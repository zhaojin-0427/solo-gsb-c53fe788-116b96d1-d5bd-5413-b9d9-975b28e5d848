"""SQLite persistence for imported scenarios and saved plans.

A *scenario* keeps the exact imported JSON (input snapshot) together with the
safety parameters used for the most recent solve, so saved plans can always be
reopened and compared against their original input.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Serializes writes inside one process; cross-process arbitration is handled
# by SQLite itself (WAL + a ``BEGIN IMMEDIATE`` write transaction).
_LOCK = threading.Lock()

#: How long SQLite waits on another process' write lock before raising
#: "database is locked"; the submit path additionally retries as a safety net.
_BUSY_TIMEOUT_MS = 5000
_BUSY_ATTEMPTS = 60
_BUSY_WAIT_SECONDS = 0.05

#: Optional pre-transaction delay for cross-process concurrency checks.
#: Reads ``DCC_GUARDED_BATCH_DELAY`` (seconds); 0/unset means no delay.  When
#: set on ONE of several server processes sharing the database, that
#: process's batch creation pauses just BEFORE opening its
#: ``BEGIN IMMEDIATE`` transaction, giving a concurrent POST .../solve on a
#: peer a head start so the "solve commits first" interleaving actually shows
#: up in an HTTP-level race.  (A batch write is otherwise a few fast SQL
#: statements and almost always beats a CP-SAT solve that must start up and
#: run.)  The delay is *outside* the write lock, so it cannot block the peer;
#: once the transaction opens it still waits for the peer to finish and then
#: reads authoritative state - correctness is unchanged.  Never set in normal
#: operation.
def _guarded_batch_pre_delay() -> float:
    return float(os.environ.get("DCC_GUARDED_BATCH_DELAY", "0") or 0.0)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL DEFAULT '',
    input_json      TEXT NOT NULL,
    safety_radius   REAL NOT NULL,
    safety_height   REAL NOT NULL,
    created_at      REAL NOT NULL,
    solved_at       REAL,
    result_json     TEXT,
    -- State of the most recent /solve run, persisted SEPARATELY from the
    -- retained (old) result row:
    --   * NULL                     -> the stored result (if any) is the
    --                                product of the latest verified solve
    --                                (feasible/infeasible); normal creation;
    --   * 'unknown'                -> the latest solve proved NOTHING (timed
    --                                out).  The stored feasible table (kept
    --                                in result_json) is an unverified OLD
    --                                table; batches may freeze it only via an
    --                                explicit, threshold-checked selection.
    -- ``unverified_solve_json`` remembers that unproven run's attempted
    -- thresholds/locks so the marker survives restarts and page reloads and
    -- can be shown alongside the retained old table.  A new verified solve
    -- (feasible or proven infeasible) clears both columns.
    latest_solve_status     TEXT,
    unverified_solve_json   TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id     INTEGER NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    name            TEXT NOT NULL DEFAULT '',
    input_json      TEXT NOT NULL,
    safety_radius   REAL NOT NULL,
    safety_height   REAL NOT NULL,
    plan_json       TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_reports (
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    sortie_id       TEXT NOT NULL,
    actual_offset   INTEGER NOT NULL CHECK (actual_offset >= 0),
    revision        INTEGER NOT NULL CHECK (revision >= 1),
    reported_at     REAL NOT NULL,
    PRIMARY KEY (batch_id, sortie_id)
);

-- One row per *accepted* report write ("node").  Each node stores the FULL
-- report state immediately before and immediately after its write, so a
-- warning-change review is a pure replay of two committed snapshots.  The
-- node row is inserted in the same BEGIN IMMEDIATE transaction as the report
-- write it describes; therefore, even with several processes sharing the
-- file, report writes and review nodes correspond atomically and the per-batch
-- seq order is exactly the database commit order.  A NULL trigger marks the
-- baseline node (an empty state for new batches, or the latest state visible
-- at upgrade time for batches created before this feature existed).
CREATE TABLE IF NOT EXISTS batch_report_nodes (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id                INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    seq                     INTEGER NOT NULL,
    trigger_sortie_id       TEXT,
    trigger_actual_offset   INTEGER,
    trigger_revision        INTEGER,
    before_json             TEXT NOT NULL,
    after_json              TEXT NOT NULL,
    created_at              REAL NOT NULL,
    UNIQUE (batch_id, seq)
);
"""


class BatchCreateRejected(Exception):
    """Batch creation refused against the authoritative in-txn state.

    ``status`` is the HTTP status the route must return and ``reason`` the
    machine-readable code; any extra fields are echoed verbatim inside the
    response ``detail``.  Raising inside the guarded write transaction rolls
    the batch insert back, so a refusal never leaves a batch behind.
    """

    def __init__(self, status: int, reason: str, **detail):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.detail = detail


#: The stored plan must be feasible before any batch can freeze it.
_NO_FEASIBLE_PLAN = "a feasible plan must be solved before creating an execution batch"


class Store:
    def __init__(self, path: str | Path = "data/console.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Enable WAL and create the schema, tolerating a peer that is doing
        the same initialization concurrently.

        Switching journal mode (and, on a fresh file, creating the tables)
        needs an exclusive lock, so a second process booting at the same time
        can briefly hit ``database is locked``; retry until it succeeds or
        confirms the schema already exists.
        """
        for attempt in range(_BUSY_ATTEMPTS):
            with self._conn() as c:
                try:
                    # WAL lets multiple processes read while one writes and
                    # keeps short write transactions from colliding.
                    c.execute("PRAGMA journal_mode=WAL")
                    c.execute("PRAGMA synchronous=NORMAL")
                    c.executescript(_SCHEMA)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or attempt == _BUSY_ATTEMPTS - 1:
                        raise
            time.sleep(_BUSY_WAIT_SECONDS)
        self._migrate_scenarios()
        # Backfill review baselines for batches that existed before the node
        # table.  Existing batches only get a single baseline node describing
        # the newest state knowable at upgrade time - earlier history is never
        # fabricated.  Idempotent and serialized via the write transaction.
        self._backfill_baselines()

    @staticmethod
    def _has_column(c, table: str, column: str) -> bool:
        return any(r["name"] == column
                   for r in c.execute(f"PRAGMA table_info({table})").fetchall())

    def _migrate_scenarios(self) -> None:
        """Add the solve-state columns to databases created before them."""
        with self._write_tx() as c:
            if not self._has_column(c, "scenarios", "latest_solve_status"):
                c.execute("ALTER TABLE scenarios ADD COLUMN latest_solve_status TEXT")
            if not self._has_column(c, "scenarios", "unverified_solve_json"):
                c.execute("ALTER TABLE scenarios ADD COLUMN unverified_solve_json TEXT")

    @staticmethod
    def _snapshot(c, batch_id: int) -> str:
        """Full per-batch report state as deterministic JSON (sortie id sorted)."""
        rows = c.execute(
            "SELECT sortie_id,actual_offset,revision FROM batch_reports "
            "WHERE batch_id=? ORDER BY sortie_id",
            (batch_id,)).fetchall()
        return json.dumps([dict(r) for r in rows], ensure_ascii=False,
                          sort_keys=True)

    def _backfill_baselines(self) -> None:
        with self._write_tx() as c:
            rows = c.execute(
                "SELECT b.id FROM batches b WHERE NOT EXISTS ("
                "SELECT 1 FROM batch_report_nodes n WHERE n.batch_id = b.id)"
            ).fetchall()
            now = time.time()
            for r in rows:
                bid = r["id"]
                snap = self._snapshot(c, bid)
                c.execute(
                    "INSERT INTO batch_report_nodes"
                    "(batch_id,seq,trigger_sortie_id,trigger_actual_offset,"
                    "trigger_revision,before_json,after_json,created_at) "
                    "VALUES (?,1,NULL,NULL,NULL,?,?,?)",
                    (bid, snap, snap, now))

    def _conn(self):
        conn = sqlite3.connect(
            self.path, timeout=_BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    @contextmanager
    def _write_tx(self):
        """A single ``BEGIN IMMEDIATE`` write transaction.

        This acquires SQLite's reserved lock up front, blocks on a writer that
        already holds it via ``busy_timeout``, and groups the conditional
        write + authoritative re-read in one commit.  It does *not* by itself
        make a read-then-write sequence safe across processes (WAL snapshots
        differ), so callers must make each write conditional in SQL.
        """
        with _LOCK, self._conn() as c:
            for attempt in range(_BUSY_ATTEMPTS):
                try:
                    c.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or attempt == _BUSY_ATTEMPTS - 1:
                        raise
                    time.sleep(_BUSY_WAIT_SECONDS)
            try:
                yield c
            except Exception:
                c.rollback()
                raise
            else:
                c.commit()

    def create_scenario(self, input_payload: dict, safety_radius: float,
                        safety_height: float, name: str = "") -> int:
        with self._write_tx() as c:
            cur = c.execute(
                "INSERT INTO scenarios(name,input_json,safety_radius,safety_height,created_at)"
                " VALUES (?,?,?,?,?)",
                (name, json.dumps(input_payload, ensure_ascii=False),
                 float(safety_radius), float(safety_height), time.time()),
            )
            return int(cur.lastrowid)

    def save_result(self, scenario_id: int, result: dict, safety_radius: float,
                    safety_height: float, name: str | None = None) -> None:
        """Persist a VERIFIED solve (feasible table or a proven infeasibility).

        The unverified/unknown marker belongs to a previous run and is cleared
        here, so the freshly proven result restores unmarked batch creation.
        """
        with self._write_tx() as c:
            if name is not None:
                c.execute("UPDATE scenarios SET name=? WHERE id=?", (name, scenario_id))
            c.execute(
                "UPDATE scenarios SET result_json=?, solved_at=?, safety_radius=?, "
                "safety_height=?, latest_solve_status=NULL, unverified_solve_json=NULL"
                " WHERE id=?",
                (json.dumps(result, ensure_ascii=False), time.time(),
                 float(safety_radius), float(safety_height), scenario_id),
            )

    def rename_scenario(self, scenario_id: int, name: str) -> None:
        """Rename a saved plan without touching solve state.

        Renaming the retained old table must not lift the unverified marker.
        """
        with self._write_tx() as c:
            c.execute("UPDATE scenarios SET name=? WHERE id=?", (name, scenario_id))

    def mark_solve_unknown(self, scenario_id: int, attempted: dict) -> None:
        """Record that the most recent solve came back ``unknown``.

        The previously stored (old) table is never overwritten; only the
        marker and the unproven run's attempted thresholds/locks are saved, so
        the distinction survives restarts, reloads and renames.  A later
        verified solve clears the marker via :meth:`save_result`.
        """
        with self._write_tx() as c:
            c.execute(
                "UPDATE scenarios SET latest_solve_status='unknown', "
                "unverified_solve_json=? WHERE id=?",
                (json.dumps(attempted, ensure_ascii=False), scenario_id),
            )

    def list_scenarios(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,name,safety_radius,safety_height,created_at,solved_at,"
                "CASE WHEN result_json IS NOT NULL THEN 1 ELSE 0 END AS has_result "
                "FROM scenarios ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scenario(self, scenario_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM scenarios WHERE id=?", (scenario_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["input"] = json.loads(d.pop("input_json"))
        d["result"] = json.loads(d["result_json"]) if d.get("result_json") else None
        d["unverified_solve"] = (
            json.loads(d.pop("unverified_solve_json"))
            if d.get("unverified_solve_json") else None)
        return d

    def delete_scenario(self, scenario_id: int) -> bool:
        with self._write_tx() as c:
            # Remove dependent batches (and their reports/nodes) explicitly so
            # the cascade works even on databases created before the FK cascade.
            c.execute("DELETE FROM batch_report_nodes WHERE batch_id IN "
                      "(SELECT id FROM batches WHERE scenario_id=?)",
                      (scenario_id,))
            c.execute("DELETE FROM batch_reports WHERE batch_id IN "
                      "(SELECT id FROM batches WHERE scenario_id=?)",
                      (scenario_id,))
            c.execute("DELETE FROM batches WHERE scenario_id=?", (scenario_id,))
            cur = c.execute("DELETE FROM scenarios WHERE id=?", (scenario_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Execution batches (frozen plan) and actual-take-off reports
    # ------------------------------------------------------------------

    def create_batch(self, scenario_id: int, name: str, input_payload: dict,
                     safety_radius: float, safety_height: float,
                     plan: dict) -> int:
        with self._write_tx() as c:
            now = time.time()
            cur = c.execute(
                "INSERT INTO batches(scenario_id,name,input_json,safety_radius,"
                "safety_height,plan_json,created_at) VALUES (?,?,?,?,?,?,?)",
                (scenario_id, name, json.dumps(input_payload, ensure_ascii=False),
                 float(safety_radius), float(safety_height),
                 json.dumps(plan, ensure_ascii=False), now),
            )
            bid = int(cur.lastrowid)
            # Baseline node (seq 1): the empty report state before any
            # accepted write. Created in the same transaction as the batch,
            # so every batch always has a replay chain starting at seq 1.
            c.execute(
                "INSERT INTO batch_report_nodes"
                "(batch_id,seq,trigger_sortie_id,trigger_actual_offset,"
                "trigger_revision,before_json,after_json,created_at) "
                "VALUES (?,1,NULL,NULL,NULL,'[]','[]',?)",
                (bid, now))
            return bid

    def create_guarded_batch(self, scenario_id: int, name: str,
                             use_stored_plan: bool,
                             confirmed_radius: float | None,
                             confirmed_height: float | None) -> int:
        """Adjudicate AND insert a batch inside ONE ``BEGIN IMMEDIATE`` txn.

        The whole point is that the eligibility read, the unknown-marker /
        original-threshold adjudication, the snapshot freeze and the batch
        insert happen against the *same* committed state, holding SQLite's
        reserved write lock the entire time.  A concurrent
        ``save_result`` (a verified feasible/infeasible re-solve) or
        ``mark_solve_unknown`` from another process therefore commits either
        strictly before this transaction's authoritative read (the batch is
        then decided on the NEW state) or strictly after its commit (the
        batch keeps the snapshot that was valid at decision time) - the
        stale "read the old table/marker outside a write txn, insert in a
        later txn" window no longer exists:

        * an ``unknown`` marker committed first is seen here, so a request
          that did not select the retained old table (or fails the original
          threshold confirmation) is rejected with no batch;
        * a new verified solve committed first is seen here too, so the
          batch freezes the NEW table/thresholds rather than a stale old
          table fetched outside the transaction.

        Raises :class:`BatchCreateRejected` (404 scenario missing, 400 no
        feasible table, 409 marker-related) and rolls back; returns the new
        batch id on success.
        """
        tol = 1e-9

        def reject(status, reason, **detail):
            raise BatchCreateRejected(status, reason, **detail)

        # Optional, env-gated pacing delay for the cross-process verification
        # script (see ``_guarded_batch_pre_delay``); 0 in normal operation.
        pre_delay = _guarded_batch_pre_delay()
        if pre_delay > 0:
            time.sleep(pre_delay)

        with self._write_tx() as c:
            row = c.execute("SELECT * FROM scenarios WHERE id=?",
                            (scenario_id,)).fetchone()
            if row is None:
                reject(404, "scenario not found")
            input_payload = json.loads(row["input_json"])
            result = json.loads(row["result_json"]) if row["result_json"] else None
            old_R, old_H = row["safety_radius"], row["safety_height"]
            if result is None or not result.get("feasible"):
                reject(400, _NO_FEASIBLE_PLAN)
            if row["latest_solve_status"] is not None:
                # The persisted unknown marker is authoritative AT THIS
                # SNAPSHOT - it can no longer arrive in the gap between the
                # route's read and the insert (any such write waits on this
                # transaction and is adjudicated by a LATER request).
                if not use_stored_plan:
                    reject(409, "latest_solve_unverified",
                           latest_solve_status="unknown",
                           stored_plan_original_safety_radius=old_R,
                           stored_plan_original_safety_height=old_H,
                           message="the most recent solve returned unknown; the "
                                   "stored feasible table is an unverified OLD "
                                   "table. Re-solve successfully first, or "
                                   "explicitly select it with use_stored_plan=true "
                                   "and confirm its original thresholds")
                if confirmed_radius is None or confirmed_height is None:
                    reject(409, "stored_plan_threshold_confirmation_required",
                           stored_plan_original_safety_radius=old_R,
                           stored_plan_original_safety_height=old_H)
                if (abs(float(confirmed_radius) - float(old_R)) > tol
                        or abs(float(confirmed_height) - float(old_H)) > tol):
                    reject(409, "stored_plan_threshold_mismatch",
                           stored_plan_original_safety_radius=old_R,
                           stored_plan_original_safety_height=old_H,
                           confirmed_safety_radius=confirmed_radius,
                           confirmed_safety_height=confirmed_height)
            # Freeze exactly the state read inside this transaction; a peer
            # write that commits afterwards belongs to later requests/batches.
            plan = {"offsets": dict(result["offsets"])}
            now = time.time()
            cur = c.execute(
                "INSERT INTO batches(scenario_id,name,input_json,safety_radius,"
                "safety_height,plan_json,created_at) VALUES (?,?,?,?,?,?,?)",
                (scenario_id, name, json.dumps(input_payload, ensure_ascii=False),
                 float(old_R), float(old_H),
                 json.dumps(plan, ensure_ascii=False), now),
            )
            bid = int(cur.lastrowid)
            c.execute(
                "INSERT INTO batch_report_nodes"
                "(batch_id,seq,trigger_sortie_id,trigger_actual_offset,"
                "trigger_revision,before_json,after_json,created_at) "
                "VALUES (?,1,NULL,NULL,NULL,'[]','[]',?)",
                (bid, now))
            return bid

    def list_batches(self, scenario_id: int | None = None) -> list[dict]:
        sql = ("SELECT b.id,b.scenario_id,b.name,b.safety_radius,b.safety_height,"
               "b.created_at, COUNT(r.sortie_id) AS reported_count "
               "FROM batches b LEFT JOIN batch_reports r "
               "ON r.batch_id = b.id ")
        args: tuple = ()
        if scenario_id is not None:
            sql += "WHERE b.scenario_id=? "
            args = (scenario_id,)
        sql += "GROUP BY b.id ORDER BY b.id DESC"
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def get_batch(self, batch_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["input"] = json.loads(d.pop("input_json"))
            d["plan"] = json.loads(d.pop("plan_json"))
            reports = c.execute(
                "SELECT sortie_id,actual_offset,revision,reported_at "
                "FROM batch_reports WHERE batch_id=? ORDER BY sortie_id",
                (batch_id,),
            ).fetchall()
        d["reports"] = [dict(r) for r in reports]
        return d

    def delete_batch(self, batch_id: int) -> bool:
        with self._write_tx() as c:
            cur = c.execute("DELETE FROM batches WHERE id=?", (batch_id,))
            return cur.rowcount > 0

    def submit_report(self, batch_id: int, sortie_id: str,
                      actual_offset: int, revision: int) -> tuple[str, dict | None]:
        """Atomically apply a revision-controlled report.

        The decision is enforced by the database itself rather than by a
        read-then-write sequence, so it stays correct when several *processes*
        share the SQLite file.  Inside one ``BEGIN IMMEDIATE`` transaction two
        guarded statements are issued:

        1. an ``INSERT ... SELECT ... WHERE NOT EXISTS(...) AND revision = 1``
           that creates the row only for a first report at revision 1;
        2. an ``UPDATE ... WHERE incoming_revision > stored_revision`` that
           replaces the row only for a strictly higher revision.

        Because every write is conditional on the row committed at write
        time, a lower revision can never overwrite a higher one and a
        same-revision write can never change an existing value, no matter how
        overlapping transactions commit.  The row is then re-read inside the
        same transaction to classify the outcome against the authoritative
        state (so a loser reports the winner's ``current_revision``):

        * no row + revision 1          -> ``inserted``
        * no row + revision != 1       -> ``invalid_first_revision``
        * revision == stored revision  -> same value ``idempotent`` /
          different value ``revision_value_conflict``
        * revision  < stored revision  -> ``stale_revision``
        * revision  > stored revision  -> ``inserted``

        Returns ``(outcome, info)``.
        """
        with self._write_tx() as c:
            now = time.time()
            # The review node must describe the exact state transition
            # performed by THIS committed transaction, so the "before"
            # snapshot is taken inside it (after BEGIN IMMEDIATE, all prior
            # writers have committed) and the node is written in the same
            # commit as the report write.
            before_snapshot = self._snapshot(c, batch_id)
            # 1) First report only: create the row when none exists AND the
            #    incoming revision is 1.
            cur = c.execute(
                "INSERT INTO batch_reports(batch_id,sortie_id,actual_offset,"
                "revision,reported_at) "
                "SELECT ?,?,?,?,? WHERE ? = 1 AND NOT EXISTS ("
                "SELECT 1 FROM batch_reports WHERE batch_id=? AND sortie_id=?)",
                (batch_id, sortie_id, int(actual_offset), int(revision), now,
                 int(revision), batch_id, sortie_id),
            )
            wrote = cur.rowcount > 0
            # 2) Higher revision only: replace an existing strictly-lower row.
            if not wrote:
                cur = c.execute(
                    "UPDATE batch_reports SET actual_offset=?,revision=?,"
                    "reported_at=? WHERE batch_id=? AND sortie_id=? AND revision < ?",
                    (int(actual_offset), int(revision), now,
                     batch_id, sortie_id, int(revision)),
                )
                wrote = cur.rowcount > 0
            if wrote:
                after_snapshot = self._snapshot(c, batch_id)
                # seq follows the per-batch node count; committed
                # transactions serialise on the write lock, so this order is
                # exactly the actual submission (commit) order, even across
                # processes sharing the file.
                seq = int(c.execute(
                    "SELECT COUNT(*) AS n FROM batch_report_nodes WHERE batch_id=?",
                    (batch_id,)).fetchone()["n"]) + 1
                c.execute(
                    "INSERT INTO batch_report_nodes"
                    "(batch_id,seq,trigger_sortie_id,trigger_actual_offset,"
                    "trigger_revision,before_json,after_json,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (batch_id, seq, sortie_id, int(actual_offset),
                     int(revision), before_snapshot, after_snapshot, now))
                return "inserted", {"sortie_id": sortie_id,
                                    "actual_offset": int(actual_offset),
                                    "revision": int(revision)}
            row = c.execute(
                "SELECT actual_offset,revision FROM batch_reports "
                "WHERE batch_id=? AND sortie_id=?",
                (batch_id, sortie_id),
            ).fetchone()
            if row is None:
                # No row exists, so the only legal write is a first report at
                # revision 1; anything else is rejected unchanged.
                return "invalid_first_revision", None
            cur_off, cur_rev = row["actual_offset"], row["revision"]
            if revision == cur_rev:
                if actual_offset != cur_off:
                    return "revision_value_conflict", {
                        "stored_actual_offset": cur_off}
                return "idempotent", {"sortie_id": sortie_id,
                                      "actual_offset": cur_off,
                                      "revision": cur_rev}
            return "stale_revision", {"current_revision": cur_rev}

    # ------------------------------------------------------------------
    # Report nodes (accepted-write events) for warning-change reviews
    # ------------------------------------------------------------------

    @staticmethod
    def _node_row(r) -> dict:
        return {
            "id": r["id"],
            "batch_id": r["batch_id"],
            "seq": r["seq"],
            "created_at": r["created_at"],
            "trigger_sortie_id": r["trigger_sortie_id"],
            "trigger_actual_offset": r["trigger_actual_offset"],
            "trigger_revision": r["trigger_revision"],
            "is_baseline": r["trigger_sortie_id"] is None,
            "before": json.loads(r["before_json"]),
            "after": json.loads(r["after_json"]),
        }

    def list_report_nodes(self, batch_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM batch_report_nodes WHERE batch_id=? ORDER BY seq",
                (batch_id,)).fetchall()
        return [self._node_row(r) for r in rows]

    def get_report_node(self, batch_id: int, node_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM batch_report_nodes WHERE batch_id=? AND id=?",
                (batch_id, node_id)).fetchone()
        return self._node_row(row) if row is not None else None
