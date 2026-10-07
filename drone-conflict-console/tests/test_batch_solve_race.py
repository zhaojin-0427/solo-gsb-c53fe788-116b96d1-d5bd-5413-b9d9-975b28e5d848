"""Interleaving guard: batch creation vs. a concurrent /solve write.

Regression test for the stale adjudication that occurred when several
*processes* shared one SQLite file:

* POST .../batches read the feasible old table and the (absent) unknown
  marker OUTSIDE a write transaction and only later opened a new transaction
  to insert the batch;
* in the gap, POST .../solve on another process could either
  - commit an ``unknown`` marker (``mark_solve_unknown``) for new
    locks/thresholds, after which a request that never selected the old
    table wrongly returned 201 (wrong release), or
  - commit a NEW verified feasible solve (``save_result``), after which the
    batch wrongly froze the stale old table/thresholds read earlier
    (wrong freeze).

The fix reads the authoritative scenario state INSIDE the single
``BEGIN IMMEDIATE`` transaction that inserts the batch, so a solve write
commits either strictly before the batch decision or strictly after it.

This file checks:

* DETERMINISTIC cross-process orderings (a multiprocessing Event pins which
  side commits first):
  - unknown committed first: an unflagged guarded batch is rejected (409),
    while a worker re-enacting the OLD read-outside-txn sequence wrongly
    inserts (an explicit A/B proving the test really exercises the hole);
  - unknown committed first + explicit ``use_stored_plan`` with the ORIGINAL
    thresholds confirmed: the old snapshot is kept (201); new attempted
    thresholds confirmed instead: mismatch (409);
  - a new verified feasible solve committed first: the batch freezes the NEW
    table/thresholds (the old sequence would freeze the stale old table);
  - batch committed first: it keeps the then-valid snapshot; the marker only
    restricts LATER requests;
  - a bare unknown with no feasible table still yields 400 and no batch;
* true simultaneous cross-process races over many trials: every observable
  outcome is a valid linearization (an unflagged 201 always freezes a
  snapshot that was the committed scenario state at its commit point; a
  marker-ahead outcome is always a 409 with no row).

Run:  python tests/test_batch_solve_race.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.storage import BatchCreateRejected, Store  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)
    else:
        print("ok:", msg)


# A feasible old table solved under R/H = 30/10.
OLD_R, OLD_H = 30.0, 10.0
NEW_R, NEW_H = 45.0, 12.0
OLD_OFFSETS = {"A": 4, "B": 0}
NEW_OFFSETS = {"A": 4, "B": 5}
SNAPSHOT = {"sorties": [{"id": "A"}, {"id": "B"}]}
UNKNOWN_ATTEMPT = {
    "status": "unknown",
    "feasible": False,
    "safety_radius": NEW_R,
    "safety_height": NEW_H,
    "locks": {"B": 5},
    "elapsed_seconds": 0.0001,
}

# Ordering modes shared by the workers:
#   "race"        -> start together, no event coordination
#   "wait_peer"   -> wait for the shared event BEFORE doing the write
#   "signal_after" -> set the shared event AFTER committing the write
RACE, WAIT, SIGNAL = "race", "wait_peer", "signal_after"


def _feasible(offsets, status="optimal"):
    return {"feasible": True, "status": status, "weighted_delay": 0,
            "offsets": dict(offsets), "locked": {}, "elapsed_seconds": 0.01,
            "solved_conflicts": [], "cores": None, "truncated": False}


# ---------------------------------------------------------------------------
# Worker processes (top-level so fork can pickle them)
# ---------------------------------------------------------------------------


def _guarded_worker(path, sid, name, use_stored, confirm,
                    start, event, mode, queue):
    try:
        store = Store(path)
        start.wait()
        if mode == WAIT:
            event.wait(30)
        bid = store.create_guarded_batch(
            sid, name, use_stored,
            confirm[0] if confirm else None,
            confirm[1] if confirm else None)
        if mode == SIGNAL:
            event.set()
        queue.put(("ok", bid))
    except BatchCreateRejected as exc:
        if mode == SIGNAL:
            event.set()
        queue.put(("reject", exc.status, exc.reason))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _old_style_batch_worker(path, sid, start, event, mode, queue):
    """Re-enact the OLD route sequence from a fresh process.

    Read the feasible old table / marker state OUTSIDE any write txn, let the
    peer commit in the gap, then open a SEPARATE transaction that inserts the
    batch straight from that stale row - exactly the buggy route.
    """
    try:
        store = Store(path)
        row = store.get_scenario(sid)  # pre-read outside the write txn
        start.wait()
        event.wait(30)
        result = row["result"]
        if result is None or not result.get("feasible"):
            queue.put(("reject", 400, "no feasible plan"))
            return
        plan = {"offsets": dict(result["offsets"])}
        bid = store.create_batch(
            sid, "old-style", row["input"], row["safety_radius"],
            row["safety_height"], plan)
        queue.put(("ok", bid))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _unknown_worker(path, sid, start, event, mode, queue):
    try:
        store = Store(path)
        start.wait()
        if mode == WAIT:
            event.wait(30)
        store.mark_solve_unknown(sid, UNKNOWN_ATTEMPT)
        if mode == SIGNAL:
            event.set()
        queue.put(("ok", None))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _feasible_worker(path, sid, start, event, mode, queue):
    try:
        store = Store(path)
        start.wait()
        if mode == WAIT:
            event.wait(30)
        store.save_result(sid, _feasible(NEW_OFFSETS), NEW_R, NEW_H)
        if mode == SIGNAL:
            event.set()
        queue.put(("ok", None))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _run_pair(ctx, w1, w2, timeout=30):
    """Run two workers ``(target, args, mode)`` behind one start barrier.

    One shared Event coordinates the order: the SIGNAL-side worker sets it
    after its commit, the WAIT-side blocks on it before its write.
    """
    start = ctx.Barrier(2)
    event = ctx.Event()
    queue = ctx.Queue()
    target1, args1, mode1 = w1
    target2, args2, mode2 = w2
    procs = [
        ctx.Process(target=target1, args=args1 + (start, event, mode1, queue)),
        ctx.Process(target=target2, args=args2 + (start, event, mode2, queue)),
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=timeout) for _ in procs]
    for p in procs:
        p.join(10)
        check(p.exitcode == 0, f"worker process finished cleanly: {p.exitcode}")
    return results


# ---------------------------------------------------------------------------
# Deterministic cross-process orderings
# ---------------------------------------------------------------------------


def _scenario_with_old_table(store):
    sid = store.create_scenario(SNAPSHOT, OLD_R, OLD_H)
    store.save_result(sid, _feasible(OLD_OFFSETS), OLD_R, OLD_H)
    return sid


def test_deterministic_unknown_first(ctx, db):
    setup = Store(db)

    # A/B on the SAME forced order (the unknown marker commits before the
    # batch txn): the OLD read-then-insert sequence wrongly freezes the
    # unverified table without any flag...
    sid = _scenario_with_old_table(setup)
    old_results = _run_pair(ctx,
        (_old_style_batch_worker, (db, sid), WAIT),
        (_unknown_worker, (db, sid), SIGNAL))
    old_bids = [r[1] for r in old_results if r[0] == "ok" and r[1] is not None]
    check(len(old_bids) == 1 and bool(setup.list_batches(sid)),
          "A/B: OLD read-outside-txn sequence wrongly creates the batch after "
          "the unknown marker committed (the hole being fixed)")
    setup.delete_batch(old_bids[0])

    # ...while the NEW guarded transaction reads the marker inside the write
    # lock and rejects the unflagged request with no batch.
    sid2 = _scenario_with_old_table(setup)
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid2, "reject", False, None), WAIT),
        (_unknown_worker, (db, sid2), SIGNAL))
    check(sorted(results) == [("ok", None),
                              ("reject", 409, "latest_solve_unverified")],
          f"unknown first: unflagged guarded batch -> 409, got {results}")
    check(setup.list_batches(sid2) == [],
          "unknown first: rejection leaves no batch")
    row = setup.get_scenario(sid2)
    check(row["unverified_solve"] is not None
          and row["safety_radius"] == OLD_R and row["safety_height"] == OLD_H,
          "unknown first: marker persisted, old table/thresholds retained")

    # Explicit selection + ORIGINAL-threshold confirmation keeps the retained
    # old snapshot (which is exactly what is valid at the batch's commit point).
    sid3 = _scenario_with_old_table(setup)
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid3, "kept", True, (OLD_R, OLD_H)), WAIT),
        (_unknown_worker, (db, sid3), SIGNAL))
    bids = [r[1] for r in results if r[0] == "ok" and r[1] is not None]
    check(len(bids) == 1 and not [r for r in results if r[0] == "reject"],
          f"unknown first: explicit confirmed batch created, got {results}")
    batch = setup.get_batch(bids[0])
    check(batch["plan"]["offsets"] == OLD_OFFSETS
          and batch["safety_radius"] == OLD_R and batch["safety_height"] == OLD_H,
          "unknown first: explicit batch freezes the old table at OLD thresholds")

    # Flag present but the NEW attempted thresholds confirmed -> mismatch.
    sid4 = _scenario_with_old_table(setup)
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid4, "mismatch", True, (NEW_R, NEW_H)), WAIT),
        (_unknown_worker, (db, sid4), SIGNAL))
    check(("reject", 409, "stored_plan_threshold_mismatch") in results,
          f"unknown first: confirming the new thresholds -> mismatch, got {results}")
    check(setup.list_batches(sid4) == [], "mismatch leaves no batch")


def test_deterministic_feasible_first(ctx, db):
    setup = Store(db)

    # A/B: the OLD sequence reads the old table, the NEW verified solve then
    # commits, and its stale insert freezes the OLD offsets/thresholds.
    sid = _scenario_with_old_table(setup)
    old_results = _run_pair(ctx,
        (_old_style_batch_worker, (db, sid), WAIT),
        (_feasible_worker, (db, sid), SIGNAL))
    old_bid = [r[1] for r in old_results if r[0] == "ok" and r[1] is not None][0]
    stale = setup.get_batch(old_bid)
    check(stale["plan"]["offsets"] == OLD_OFFSETS
          and stale["safety_radius"] == OLD_R,
          "A/B: OLD sequence freezes the STALE old table even though a new "
          "verified solve committed before its insert (the hole being fixed)")
    setup.delete_batch(old_bid)

    # New path: the in-txn read sees the committed NEW solve -> NEW table.
    sid2 = _scenario_with_old_table(setup)
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid2, "new", False, None), WAIT),
        (_feasible_worker, (db, sid2), SIGNAL))
    bids = [r[1] for r in results if r[0] == "ok" and r[1] is not None]
    check(len(bids) == 1, f"feasible first: batch created, got {results}")
    batch = setup.get_batch(bids[0])
    check(batch["plan"]["offsets"] == NEW_OFFSETS
          and batch["safety_radius"] == NEW_R and batch["safety_height"] == NEW_H,
          "feasible first: batch freezes the NEW table/thresholds committed "
          "ahead of it (no stale old-table freeze)")


def test_deterministic_batch_first(ctx, db):
    setup = Store(db)
    sid = _scenario_with_old_table(setup)

    # The batch transaction commits first; the unknown worker only writes
    # AFTER the batch signals its commit.
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid, "first", False, None), SIGNAL),
        (_unknown_worker, (db, sid), WAIT))
    bids = [r[1] for r in results if r[0] == "ok" and r[1] is not None]
    check(len(bids) == 1, f"batch first: 201, got {results}")
    batch = setup.get_batch(bids[0])
    check(batch["plan"]["offsets"] == OLD_OFFSETS
          and batch["safety_radius"] == OLD_R and batch["safety_height"] == OLD_H,
          "batch first: it retains the snapshot valid when it committed")
    # The marker landed afterwards and governs LATER requests only.
    check(setup.get_scenario(sid)["unverified_solve"] is not None,
          "batch first: unknown marker landed after the committed batch")
    try:
        setup.create_guarded_batch(sid, "later", False, None, None)
        raised = None
    except BatchCreateRejected as exc:
        raised = exc
    check(raised is not None and raised.status == 409
          and raised.reason == "latest_solve_unverified",
          "batch first: a LATER unflagged request is now restricted")
    check(len(setup.list_batches(sid)) == 1,
          "batch first: the later refusal adds no batch")


def test_deterministic_bare_unknown(ctx, db):
    setup = Store(db)
    sid = setup.create_scenario(SNAPSHOT, NEW_R, NEW_H)  # no stored result
    results = _run_pair(ctx,
        (_guarded_worker, (db, sid, "bare", False, None), WAIT),
        (_unknown_worker, (db, sid), SIGNAL))
    check(("reject", 400,
           "a feasible plan must be solved before creating an execution batch")
          in results, f"bare unknown first -> 400, got {results}")
    check(setup.list_batches(sid) == [], "bare unknown: no batch left")


# ---------------------------------------------------------------------------
# True simultaneous cross-process races
# ---------------------------------------------------------------------------


def test_true_unknown_races(ctx, db, trials=40):
    setup = Store(db)
    batch_first = unknown_first = 0
    for trial in range(trials):
        sid = _scenario_with_old_table(setup)
        results = _run_pair(ctx,
            (_guarded_worker, (db, sid, f"r{trial}", False, None), RACE),
            (_unknown_worker, (db, sid), RACE))
        check(all(r[0] != "error" for r in results),
              f"trial {trial}: no worker errors, got {results}")
        rows = setup.list_batches(sid)
        marker = setup.get_scenario(sid)["unverified_solve"]
        created = [r for r in results if r[0] == "ok" and r[1] is not None]
        rejected = [r for r in results if r[0] == "reject"]
        if created:
            batch_first += 1
            check(not rejected and marker is not None and len(rows) == 1,
                  f"trial {trial}: batch-first shape, got {results}")
            batch = setup.get_batch(created[0][1])
            check(batch["plan"]["offsets"] == OLD_OFFSETS
                  and batch["safety_radius"] == OLD_R
                  and batch["safety_height"] == OLD_H,
                  f"trial {trial}: batch-first retains the valid old snapshot")
        else:
            unknown_first += 1
            check(len(rows) == 0 and marker is not None
                  and rejected[0][1] == 409
                  and rejected[0][2] == "latest_solve_unverified",
                  f"trial {trial}: unknown-first shape, got {results}")
    print(f"  (true unknown races: batch-first={batch_first}, "
          f"unknown-first={unknown_first} over {trials} trials)")
    check(batch_first > 0 and unknown_first > 0,
          "both commit orders actually exercised (the barrier race is real)")


def test_true_feasible_races(ctx, db, trials=40):
    """Both linearizations may win; the frozen snapshot must never be mixed."""
    setup = Store(db)
    orders = {"old": 0, "new": 0}
    for trial in range(trials):
        sid = _scenario_with_old_table(setup)
        results = _run_pair(ctx,
            (_guarded_worker, (db, sid, f"f{trial}", False, None), RACE),
            (_feasible_worker, (db, sid), RACE))
        check(all(r[0] == "ok" for r in results),
              f"trial {trial}: both writes succeed, got {results}")
        bid = [r[1] for r in results if isinstance(r[1], int)][0]
        batch = setup.get_batch(bid)
        offs = batch["plan"]["offsets"]
        R, H = batch["safety_radius"], batch["safety_height"]
        if (offs, R, H) == (OLD_OFFSETS, OLD_R, OLD_H):
            orders["old"] += 1
        elif (offs, R, H) == (NEW_OFFSETS, NEW_R, NEW_H):
            orders["new"] += 1
        else:
            check(False, f"trial {trial}: impossible mixed snapshot "
                         f"offsets={offs} R={R} H={H}")
        check(batch["input"] == SNAPSHOT, f"trial {trial}: input snapshot frozen")
    print(f"  (true feasible races: batch-first(old)={orders['old']}, "
          f"solve-first(new)={orders['new']} over {trials} trials)")
    check(orders["old"] > 0 and orders["new"] > 0,
          "both linearizations really occur under the simultaneous race")


def main():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        ctx = mp.get_context("fork")
        test_deterministic_unknown_first(ctx, db)
        test_deterministic_feasible_first(ctx, db)
        test_deterministic_batch_first(ctx, db)
        test_deterministic_bare_unknown(ctx, db)
        test_true_unknown_races(ctx, db)
        test_true_feasible_races(ctx, db)
    print(f"batch/solve interleaving checks, failures: {FAILS}")
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
