#!/usr/bin/env python3
"""Two-process shared-SQLite verification over real HTTP.

Assumes TWO server processes are running against the same SQLite database:

    BASE1 (default http://localhost:8000)  -> process 1
    BASE2 (default http://localhost:8001)  -> process 2, same DCC_DB_PATH

Easiest way (see README):

    docker compose --profile multiproc up --build
    python3 scripts/verify_multiprocess.py

The script imports the bundled example, solves/saves it, creates execution
batches and then exercises the revision protocol *across the two processes*:

* case A: revision 3 committed first, then revision 2 from the other process
  must be rejected as a stale revision whose ``current_revision`` is 3;
* case B: revision 2 committed first, revision 3 afterwards still updates;
* case C: revisions 2 and 3 fired truly simultaneously from both processes,
  repeated several times - the stored row must always end at revision 3;
* case D: two simultaneous first reports (revision 1, same value) - exactly
  one insert and one idempotent retry.
* review nodes: every accepted write above gets exactly one node (idempotent
  and rejected writes get none), visible in the same commit order from both
  processes; the node's before/after snapshots replay to the stored state and
  a review reloaded from either process returns identical first-conflict
  deltas; an unknown node returns 404.
* batch-vs-solve interleaving: POST .../batches is fired from one process at
  the same instant POST .../solve is fired from the OTHER process against the
  SAME shared SQLite file.  However the write transactions serialize, the
  observed result must be a valid linearization:
  - against a timed-out ``unknown`` solve (retained OLD table): either the
    batch commits first (201 freezing the old table, and a LATER unflagged
    request is then refused 409) or the unknown marker commits first
    (the unflagged batch is refused 409 ``latest_solve_unverified`` and NO
    batch is created; only ``use_stored_plan:true`` + the ORIGINAL thresholds
    still creates it); the forbidden stale outcome - 201 on a request whose
    decision predated a marker committed ahead of its insert - never appears;
  - against a verified feasible re-solve under a lock: the frozen batch is
    ALWAYS one internally consistent snapshot (either the old offsets at the
    old thresholds or the new offsets at the new thresholds, never mixed).

After every case it reloads the batch (and its warning) from BOTH processes
and checks they show the highest revision's actual offset. Exits non-zero on
the first failed expectation.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE1 = os.environ.get("BASE1", "http://localhost:8000")
BASE2 = os.environ.get("BASE2", "http://localhost:8001")
ROOT = Path(__file__).resolve().parent.parent

FAILURES = 0


def check(cond, msg):
    global FAILURES
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILURES += 1


def call(method, base, path, payload=None):
    """Return (http_status, parsed_json)."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def wait_healthy(base):
    for _ in range(60):
        try:
            status, _ = call("GET", base, "/api/health")
            if status == 200:
                return
        except OSError:
            pass
        import time
        time.sleep(1)
    raise SystemExit(f"server not healthy: {base}")


def setup_scenario_and_plan():
    payload = json.loads((ROOT / "examples" / "crossroads.json").read_text())
    status, body = call("POST", BASE1, "/api/import", payload)
    check(status == 200, f"import scenario via {BASE1} (HTTP {status})")
    sid = body["scenario_id"]

    status, body = call("POST", BASE1, f"/api/scenarios/{sid}/solve",
                        {"time_limit": 20})
    check(status == 200 and body["solution"]["feasible"],
          f"solve a feasible plan (HTTP {status})")
    call("POST", BASE1, f"/api/scenarios/{sid}/save",
         {"name": "multiproc-check"})
    return sid


def new_batch(sid, name):
    status, body = call("POST", BASE1, f"/api/scenarios/{sid}/batches",
                        {"name": name})
    check(status == 201, f"create batch {name!r} (HTTP {status})")
    return body["batch_id"]


def report(base, bid, sortie, offset, revision):
    return call("PUT", base, f"/api/batches/{bid}/reports",
                {"sortie_id": sortie, "actual_offset": offset,
                 "revision": revision})


def stored_report(base, bid, sortie):
    status, body = call("GET", base, f"/api/batches/{bid}")
    assert status == 200
    rows = {r["sortie_id"]: r for r in body["reports"]}
    return rows.get(sortie)


def warning_row(base, bid, t):
    status, body = call("GET", base, f"/api/batches/{bid}/warning?t={t}")
    assert status == 200, (status, body)
    return {r["id"]: r for r in body["warning"]["sorties"]}["UAV-A"]


def report_nodes(base, bid):
    status, body = call("GET", base, f"/api/batches/{bid}/report-nodes")
    assert status == 200, (status, body)
    return body["nodes"]


def review(base, bid, node_id, t):
    return call("GET", base,
                f"/api/batches/{bid}/report-nodes/{node_id}/review?t={t}")


def batches(base, sid=None):
    path = "/api/batches" + (f"?scenario_id={sid}" if sid is not None else "")
    status, body = call("GET", base, path)
    assert status == 200, (status, body)
    return body["batches"]


def solve(base, sid, payload):
    return call("POST", base, f"/api/scenarios/{sid}/solve", payload)


def create_batch_async(pool, base, sid, body=None):
    """Fire a batch creation; return (future, release_barrier).

    The caller passes the same barrier to a concurrent solve future so both
    HTTP requests are released at the same instant.
    """
    start = threading.Barrier(2)

    def go():
        start.wait()
        return call("POST", base, f"/api/scenarios/{sid}/batches",
                    body if body is not None else {"name": "race-batch"})

    return pool.submit(go), start


# BASE2 runs with DCC_GUARDED_BATCH_DELAY set (see README / docker-compose):
# its batch write pauses just BEFORE opening the write transaction, so a solve
# fired concurrently from BASE1 commits first.  Batches sent to BASE1 are
# undelayed and normally commit first.  Alternating per round exercises both
# real serialization orders deterministically.
DELAYED, FAST = BASE2, BASE1


def _race_bases(trial):
    """Return (batch_base, solve_base) with the two processes opposed.

    Even rounds put the batch on the DELAYED process (the solve on the fast
    peer commits first); odd rounds put it on the FAST process (the batch
    commits first).
    """
    if trial % 2 == 0:
        return DELAYED, FAST
    return FAST, DELAYED


def batch_vs_solve_unknown_checks(pool):
    """Unflagged batch racing an unknown solve that retains an OLD table.

    Even rounds fire the batch from the DELAYED process (the unknown marker
    commits first); odd rounds fire it from the FAST process (the batch
    commits first).
    """
    payload = json.loads((ROOT / "examples" / "crossroads.json").read_text())
    old_offsets = {"UAV-A": 4, "UAV-B": 0, "UAV-C": 2, "UAV-D": 6}
    unknown_req = {"time_limit": 0.0001, "safety_radius": 45,
                   "safety_height": 12,
                   "locks": [{"sortie_id": "UAV-B", "offset": 5}]}
    orders = {"batch_first": 0, "unknown_first": 0}
    rounds = 10
    for trial in range(rounds):
        batch_base, solve_base = _race_bases(trial)
        status, body = call("POST", BASE1, "/api/import", payload)
        assert status == 200, (status, body)
        sid = body["scenario_id"]
        status, body = solve(BASE1, sid, {"time_limit": 20})
        assert status == 200 and body["solution"]["feasible"], (status, body)
        assert batches(BASE1, sid) == []

        bf, barrier = create_batch_async(pool, batch_base, sid)

        def solve_go():
            barrier.wait()
            return solve(solve_base, sid, unknown_req)

        sf = pool.submit(solve_go)
        bstatus, bbody = bf.result()
        sstatus, sbody = sf.result()
        check(sstatus == 200 and sbody["solution"]["status"] == "unknown",
              f"U{trial}: peer solve is unknown (HTTP {sstatus})")
        rows = batches(BASE1, sid)
        _, mbody = call("GET", BASE1, f"/api/scenarios/{sid}")
        has_marker = mbody.get("unverified_solve") is not None
        if bstatus == 201:
            orders["batch_first"] += 1
            check(batch_base == FAST,
                  f"U{trial}: 201 only from the undelayed (batch-first) side")
            # The batch legitimately serialized BEFORE the marker: it freezes
            # the old snapshot, and the marker only governs later requests.
            check(has_marker and len(rows) == 1
                  and bbody["planned_offsets"] == old_offsets
                  and bbody["safety_radius"] == 30
                  and bbody["safety_height"] == 10,
                  f"U{trial}: batch-first freezes old table 30/10 -> {bbody}")
            status2, body2 = call("POST", BASE2, f"/api/scenarios/{sid}/batches",
                                  {"name": "after"})
            check(status2 == 409
                  and body2["detail"]["reason"] == "latest_solve_unverified",
                  f"U{trial}: later unflagged batch refused 409 (got {status2})")
            check(len(batches(BASE1, sid)) == 1,
                  f"U{trial}: refusal created no extra batch")
            # Explicit old-table selection with original thresholds still works.
            status2, body2 = call(
                "POST", BASE1, f"/api/scenarios/{sid}/batches",
                {"name": "explicit", "use_stored_plan": True,
                 "confirmed_safety_radius": 30,
                 "confirmed_safety_height": 10})
            check(status2 == 201 and body2["planned_offsets"] == old_offsets,
                  f"U{trial}: explicit confirmed old-table batch 201 (got {status2})")
        else:
            orders["unknown_first"] += 1
            check(batch_base == DELAYED,
                  f"U{trial}: refusal expected from the delayed (marker-first) "
                  f"side, got {bstatus}")
            # The marker committed ahead of the batch decision: the unflagged
            # request MUST be refused and leave no batch (the stale-release
            # bug would have returned 201 here).
            check(bstatus == 409
                  and bbody["detail"]["reason"] == "latest_solve_unverified"
                  and has_marker and rows == [],
                  f"U{trial}: unknown-first refuses unflagged batch, no row "
                  f"-> {bstatus} {bbody}, rows={rows}")
            # New attempted thresholds confirmed -> mismatch, still no batch.
            status2, body2 = call(
                "POST", BASE2, f"/api/scenarios/{sid}/batches",
                {"use_stored_plan": True,
                 "confirmed_safety_radius": 45,
                 "confirmed_safety_height": 12})
            check(status2 == 409
                  and body2["detail"]["reason"] == "stored_plan_threshold_mismatch"
                  and batches(BASE1, sid) == [],
                  f"U{trial}: mismatch confirmation refused, no row (got {status2})")
    print(f"  (batch vs unknown over {rounds} rounds: "
          f"batch-first={orders['batch_first']}, "
          f"unknown-first={orders['unknown_first']})")
    check(orders["batch_first"] > 0 and orders["unknown_first"] > 0,
          "batch-vs-unknown: both real commit orders were exercised")


def batch_vs_feasible_checks(pool):
    """Unflagged batch racing a verified feasible re-solve with new offsets."""
    payload = json.loads((ROOT / "examples" / "crossroads.json").read_text())
    old_offsets = {"UAV-A": 4, "UAV-B": 0, "UAV-C": 2, "UAV-D": 6}
    new_offsets = {"UAV-A": 4, "UAV-B": 5, "UAV-C": 10, "UAV-D": 11}
    # New verified solve: keep the ORIGINAL thresholds but lock UAV-B=5; this
    # is proven optimal quickly and changes the planned offsets.
    locked_req = {"time_limit": 20,
                  "locks": [{"sortie_id": "UAV-B", "offset": 5}]}
    orders = {"old_snapshot": 0, "new_snapshot": 0}
    rounds = 10
    for trial in range(rounds):
        batch_base, solve_base = _race_bases(trial)
        status, body = call("POST", BASE1, "/api/import", payload)
        assert status == 200, (status, body)
        sid = body["scenario_id"]
        status, body = solve(BASE1, sid, {"time_limit": 20})
        assert status == 200 and body["solution"]["feasible"], (status, body)
        assert batches(BASE1, sid) == []

        bf, barrier = create_batch_async(pool, batch_base, sid)

        def solve_go():
            barrier.wait()
            return solve(solve_base, sid, locked_req)

        sf = pool.submit(solve_go)
        bstatus, bbody = bf.result()
        sstatus, sbody = sf.result()
        check(sstatus == 200 and sbody["solution"]["feasible"]
              and sbody["solution"]["offsets"] == new_offsets,
              f"F{trial}: peer solve is the new feasible table (HTTP {sstatus})")
        check(bstatus == 201, f"F{trial}: batch created (HTTP {bstatus})")
        rows = batches(BASE1, sid)
        check(len(rows) == 1, f"F{trial}: exactly one batch, got {rows}")
        # Whichever transaction won, the frozen batch must be ONE consistent
        # committed snapshot (offsets and thresholds read together in-txn),
        # never a mixed freeze.
        offs = bbody["planned_offsets"]
        R, H = bbody["safety_radius"], bbody["safety_height"]
        if offs == old_offsets:
            orders["old_snapshot"] += 1
            check(batch_base == FAST and (R, H) == (30, 10),
                  f"F{trial}: old offsets frozen WITH old thresholds on the "
                  f"batch-first side, got {R}/{H}/{batch_base}")
        elif offs == new_offsets:
            orders["new_snapshot"] += 1
            check(batch_base == DELAYED and (R, H) == (30, 10),
                  f"F{trial}: solve-first side freezes the new locked offsets "
                  f"(thresholds stay 30/10), got {R}/{H}/{batch_base}")
        else:
            check(False, f"F{trial}: impossible offsets snapshot {offs}")
        # The frozen batch is visible identically from BOTH processes.
        got = {b["id"]: b for b in batches(BASE2, sid)}
        check(bbody["batch_id"] in got, f"F{trial}: batch visible via BASE2")
    print(f"  (batch vs feasible over {rounds} rounds: "
          f"batch-first(old)={orders['old_snapshot']}, "
          f"solve-first(new)={orders['new_snapshot']})")
    check(orders["old_snapshot"] > 0 and orders["new_snapshot"] > 0,
          "batch-vs-feasible: both real commit orders were exercised")


def batch_vs_rethreshold_checks(pool):
    """Batch racing a feasible re-solve under NEW, SMALLER thresholds.

    Smaller safety parameters relax constraints; the lexicographic optimum
    shifts to DIFFERENT offsets AND the stored thresholds change, so a stale
    freeze would surface as an impossible (new offsets, old thresholds) /
    (old offsets, new thresholds) mix.  Every round uses a fresh import so the
    "before" table is deterministically the original 30/10 optimum.
    """
    payload = json.loads((ROOT / "examples" / "crossroads.json").read_text())
    old_offsets = {"UAV-A": 4, "UAV-B": 0, "UAV-C": 2, "UAV-D": 6}
    new_offsets = {"UAV-A": 0, "UAV-B": 0, "UAV-C": 0, "UAV-D": 0}
    reth = {"time_limit": 20, "safety_radius": 15, "safety_height": 5}
    orders = {"old": 0, "new": 0}
    rounds = 10
    for trial in range(rounds):
        batch_base, solve_base = _race_bases(trial)
        status, body = call("POST", BASE1, "/api/import", payload)
        assert status == 200, (status, body)
        sid = body["scenario_id"]
        status, body = solve(BASE1, sid, {"time_limit": 20})
        assert status == 200 and body["solution"]["feasible"]
        assert body["solution"]["offsets"] == old_offsets, body

        bf, barrier = create_batch_async(pool, batch_base, sid)

        def solve_go():
            barrier.wait()
            return solve(solve_base, sid, reth)

        sf = pool.submit(solve_go)
        bstatus, bbody = bf.result()
        sstatus, sbody = sf.result()
        check(sstatus == 200 and sbody["solution"]["feasible"]
              and sbody["solution"]["offsets"] == new_offsets,
              f"T{trial}: re-threshold solve gives the all-zero table "
              f"(HTTP {sstatus} {sbody.get('solution', {}).get('offsets')})")
        check(bstatus == 201 and len(batches(BASE1, sid)) == 1,
              f"T{trial}: one batch created (HTTP {bstatus})")
        offs = bbody["planned_offsets"]
        R, H = bbody["safety_radius"], bbody["safety_height"]
        # Offsets and thresholds must always come from the SAME committed read,
        # and the winning side must match the forced ordering.
        if (offs, R, H) == (old_offsets, 30, 10):
            orders["old"] += 1
            check(batch_base == FAST,
                  f"T{trial}: old snapshot only from the batch-first side")
        elif (offs, R, H) == (new_offsets, 15, 5):
            orders["new"] += 1
            check(batch_base == DELAYED,
                  f"T{trial}: new snapshot only from the solve-first side")
        else:
            check(False, f"T{trial}: impossible mixed snapshot "
                         f"offsets={offs} R={R} H={H}")
    print(f"  (batch vs re-threshold over {rounds} rounds: "
          f"batch-first(old 30/10)={orders['old']}, "
          f"solve-first(new 15/5)={orders['new']})")
    check(orders["old"] > 0 and orders["new"] > 0,
          "re-threshold: both consistent snapshot orders were observed")


def race(bid, offers):
    """offers: list of (base, offset, revision); release at the same instant."""
    start = threading.Barrier(len(offers))

    def one(base, off, rev):
        start.wait()
        return report(base, bid, "UAV-A", off, rev)

    with ThreadPoolExecutor(max_workers=len(offers)) as pool:
        return list(pool.map(lambda a: one(*a), offers))


def main():
    wait_healthy(BASE1)
    wait_healthy(BASE2)
    print(f"both processes healthy: {BASE1} and {BASE2}")
    sid = setup_scenario_and_plan()
    # planned offset of UAV-A in the unique optimal plan is 4
    planned_a = 4

    # --- case A: rev 3 first, rev 2 second -> stale conflict ---------------
    bid_a = new_batch(sid, "caseA-3-then-2")
    check(report(BASE1, bid_a, "UAV-A", 7, 1)[0] == 200,
          "A: revision 1 accepted")
    check(report(BASE1, bid_a, "UAV-A", 9, 3)[0] == 200,
          "A: revision 3 accepted on process 1")
    status, body = report(BASE2, bid_a, "UAV-A", 8, 2)
    check(status == 409 and body["detail"]["reason"] == "stale_revision"
          and body["detail"]["current_revision"] == 3,
          f"A: revision 2 on process 2 -> 409 stale, current=3, got {status} {body}")

    # --- case B: rev 2 first, rev 3 second -> 3 still updates --------------
    bid_b = new_batch(sid, "caseB-2-then-3")
    check(report(BASE1, bid_b, "UAV-A", 7, 1)[0] == 200,
          "B: revision 1 accepted")
    check(report(BASE2, bid_b, "UAV-A", 8, 2)[0] == 200,
          "B: revision 2 accepted on process 2")
    status, body = report(BASE1, bid_b, "UAV-A", 9, 3)
    check(status == 200 and body["report"]["revision"] == 3,
          f"B: revision 3 on process 1 still updates (HTTP {status})")

    # Both orders must leave rev 3 / offset 9, visible from BOTH processes,
    # and the warning must use that actual offset (deviation = 9 - 4).
    def assert_final(case_name, case_bid):
        for base in (BASE1, BASE2):
            rep = stored_report(base, case_bid, "UAV-A")
            check(rep is not None and rep["revision"] == 3
                  and rep["actual_offset"] == 9,
                  f"{case_name}: {base} batch detail shows rev 3 / offset 9 -> {rep}")
        row = warning_row(BASE1, case_bid, 9)
        check(row["actual_offset"] == 9 and row["revision"] == 3
              and row["deviation_seconds"] == 9 - planned_a,
              f"{case_name}: warning uses rev 3 actual offset 9 -> {row}")

    assert_final("A", bid_a)
    assert_final("B", bid_b)

    # --- case C: simultaneous rev 2 vs rev 3, repeated ---------------------
    for trial in range(5):
        cb = new_batch(sid, f"caseC-race-{trial}")
        check(report(BASE1, cb, "UAV-A", 7, 1)[0] == 200,
              f"C{trial}: seed revision 1")
        results = race(cb, [(BASE1, 9, 3), (BASE2, 8, 2)])

        def is_for(rev, status, body):
            if status == 200:
                return body.get("report", {}).get("revision") == rev
            return body.get("detail", {}).get("received_revision") == rev

        r3 = [r for r in results if is_for(3, *r)]
        r2 = [r for r in results if is_for(2, *r)]
        check(len(r3) == 1 and len(r2) == 1 and r3[0][0] == 200,
              f"C{trial}: exactly one rev3 reply and it is accepted, got {results}")
        # rev2 either lost to a committed 3 (409 current=3) or serialized first
        # and was then overwritten by 3 - it must never be the final value.
        s2, b2 = r2[0]
        check(s2 == 200 or (b2["detail"]["reason"] == "stale_revision"
                            and b2["detail"]["current_revision"] == 3),
              f"C{trial}: rev2 allowed outcomes only, got {s2} {b2}")
        rep = stored_report(BASE1, cb, "UAV-A")
        check(rep["revision"] == 3 and rep["actual_offset"] == 9,
              f"C{trial}: final row always rev 3 / offset 9 -> {rep}")

    # --- case D: simultaneous first reports, same value --------------------
    db = new_batch(sid, "caseD-first-same")
    results = race(db, [(BASE1, 7, 1), (BASE2, 7, 1)])
    idem = sorted((s, b.get("idempotent")) for s, b in results)
    check(all(s == 200 for s, _ in idem)
          and {v for _, v in idem} == {False, True},
          f"D: concurrent identical first reports = 1 insert + 1 idempotent, "
          f"got {results}")
    rep = stored_report(BASE2, db, "UAV-A")
    check(rep is not None and rep["revision"] == 1 and rep["actual_offset"] == 7,
          f"D: single rev 1 / offset 7 row visible on process 2 -> {rep}")

    # --- review nodes across the two processes -----------------------------
    # Case A had accepted writes rev1 then rev3 (rev2 was rejected): the
    # baseline + exactly two trigger nodes, no node for the stale rev2.
    for base in (BASE1, BASE2):
        nodes = report_nodes(base, bid_a)
        triggers = [(n["trigger_sortie_id"], n["trigger_revision"],
                     n["trigger_actual_offset"]) for n in nodes if not n["is_baseline"]]
        check(nodes[0]["is_baseline"] and nodes[0]["seq"] == 1,
              f"A: node chain starts with baseline (via {base}) -> {nodes}")
        check(triggers == [("UAV-A", 1, 7), ("UAV-A", 3, 9)],
              f"A: one node per accepted write in commit order (via {base}) "
              f"-> {triggers}")

    # Case B's chain (rev1, rev2, rev3): reviewing the rev3 node at t=9 must
    # succeed from BOTH processes with identical before/after first-conflict
    # times and an idempotent reload.
    nodes_b = report_nodes(BASE1, bid_b)
    rev3_node = [n for n in nodes_b
                 if n["trigger_sortie_id"] == "UAV-A"
                 and n["trigger_revision"] == 3][0]
    reviews = {}
    for base in (BASE1, BASE2):
        status, body = review(base, bid_b, rev3_node["node_id"], 9)
        check(status == 200, f"review rev3 node via {base} (HTTP {status})")
        r = body["review"]
        check(r["node"]["sortie_id"] == "UAV-A" and r["node"]["revision"] == 3
              and r["node"]["actual_offset"] == 9,
              f"review names the triggering write (via {base})")
        check(r["warning_before"]["sorties"] and r["warning_after"]["sorties"],
              f"review restores full before/after states (via {base})")
        reviews[base] = (r["first_conflict_time_before"],
                         r["first_conflict_time_after"],
                         [(p["sortie_a"], p["sortie_b"],
                           tuple((x["change"], x["time_start"], x["time_end"])
                                 for x in p["added"] + p["removed"]))
                          for p in r["pair_changes"]])
    check(reviews[BASE1] == reviews[BASE2],
          f"review identical across processes / reloads -> {reviews}")

    # Re-review returns byte-identical content on repeat (determinism).
    s1, b1 = review(BASE1, bid_b, rev3_node["node_id"], 9)
    s2, b2 = review(BASE1, bid_b, rev3_node["node_id"], 9)
    check(s1 == s2 == 200 and b1 == b2, "repeated review is deterministic")

    # The raced case C: however rev2/rev3 committed, nodes exist only for the
    # writes that actually inserted, strictly ordered, final snapshot rev3.
    for trial in range(5):
        cb_id = new_batch(sid, f"caseC-node-{trial}")
        report(BASE1, cb_id, "UAV-A", 7, 1)
        race(cb_id, [(BASE1, 9, 3), (BASE2, 8, 2)])
        nodes = report_nodes(BASE1, cb_id)
        revs = [n["trigger_revision"] for n in nodes if not n["is_baseline"]]
        check(revs[0] == 1 and revs[-1] == 3
              and (revs == [1, 3] or revs == [1, 2, 3]),
              f"C{trial}: nodes follow actual commit order -> {revs}")
        if revs == [1, 3]:
            # rev2 lost the race: verify it really created no node (same as
            # its 409 stale response).
            check(any(n["trigger_revision"] == 2 for n in nodes) is False,
                  f"C{trial}: rejected stale rev2 left no node")
        # Every committed node replays from process 1 with both states.
        for n in nodes[1:]:
            status, body = review(BASE1, cb_id, n["node_id"], 9)
            check(status == 200
                  and "warning_before" in body.get("review", {})
                  and "warning_after" in body.get("review", {}),
                  f"C{trial}: node rev{n['trigger_revision']} review replays "
                  f"both states (HTTP {status})")
        rep = stored_report(BASE2, cb_id, "UAV-A")
        check(rep["revision"] == 3 and rep["actual_offset"] == 9,
              f"C{trial}: stored row still highest revision")

    # Unknown node -> 404; unknown batch -> 404.
    status, _ = review(BASE1, bid_b, 999999, 9)
    check(status == 404, f"unknown review node -> 404 (got {status})")
    status, _ = call("GET", BASE1, "/api/batches/999999/report-nodes")
    check(status == 404, f"unknown batch node list -> 404 (got {status})")

    # --- batch creation vs concurrent /solve across the two processes ------
    # The batch decision and snapshot freeze run in ONE BEGIN IMMEDIATE
    # transaction, so whichever of batch/solve commits first is exactly what
    # the other side sees; no stale "read outside, insert later" 201.
    with ThreadPoolExecutor(max_workers=4) as pool:
        print("-- batch vs UNKNOWN solve (old table retained) --")
        batch_vs_solve_unknown_checks(pool)
        print("-- batch vs verified feasible locked re-solve --")
        batch_vs_feasible_checks(pool)
        print("-- batch vs feasible re-solve with NEW thresholds --")
        batch_vs_rethreshold_checks(pool)

    print()
    if FAILURES:
        print(f"RESULT: {FAILURES} check(s) FAILED")
        sys.exit(1)
    print("RESULT: all two-process shared-DB checks passed")


if __name__ == "__main__":
    main()
