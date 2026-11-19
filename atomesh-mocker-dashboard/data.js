window.BENCHMARK_DATA = {
  "lastUpdate": 1795051372691,
  "repoUrl": "https://github.com/ZhiweiYan-96/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "cmq0113@163.com",
            "name": "Mengqing Cao",
            "username": "MengqingCao"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9",
          "message": "feat(pd): export decode admission metrics (#3173)\n\n* feat(pd): export decode admission and deferred-free KV metrics\n\nDecode admission state, per D rank (`atom:pd_admission_*`):\n\n- `pending_requests{state=\"receiving|ready|cancel_draining\"}` and\n  `pending_kv_bytes`: the credits held and the receive-target bytes behind\n  them.\n- `deferred_total{reason}`: requests deferred by a limit, once per request\n  and limit however many passes they wait.\n- `credit_wait_seconds`: first deferral to admission; 0 when admitted at once.\n- `oversized_total` and `ready_wait_seconds`: isolated admissions above the\n  budget, and receive completion to the first local forward.\n\nEvery scheduler also exports `atom:requests_deferred_free` and\n`atom:kv_cache_blocks_deferred_free` (physical blocks, shared prefixes\ncounted once): on P/D prefill, the source KV decode has not pulled yet. They\nare computed with the metrics snapshot, not per step.\n\n* fix(pd): address review on decode admission metrics\n\n- Count kv_blocks_deferred_free with a boolean mask over the pool instead\n  of a Python set of every block id. The snapshot runs on the engine thread\n  every push interval; the set cost 80-2500 ms at P/D backpressure.\n- Keep only the Prometheus counters for admission deferrals and oversized\n  admissions. The parallel `deferred` / `oversized_admissions` fields had no\n  reader outside tests, so the counters were never checked.\n- Register the admission counters without the `_total` suffix in the family\n  name (prometheus_client adds it on export); docs follow.\n- Describe deferred-free as whole-request holds; early-release offload leases\n  return once the source read finishes and are omitted.\n- Test the oversized path, cancel_draining, and that Scheduler passes its\n  metrics to admission.\n\n* test(pd): count deferred-free blocks from a real block table\n\nThe snapshot reads array('i') tables through np.frombuffer. The existing\ncase passed plain lists, so that branch never ran.\n\n* test(pd): a rejected deferred request leaves no admission marker\n\nThe deferral marker now lives in the controller keyed by request id, so a\nrequest rejected from waiting before admission must not leave it behind.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* refactor(pd): drop the P-side deferred-free metrics\n\nrequests_deferred_free and kv_cache_blocks_deferred_free count prefill-side\nKV that decode has not pulled. They do not depend on decode admission, and\nnothing in this PR needs them.\n\n* refactor(pd): export decode admission state from the metrics snapshot\n\n- Keep pending_requests{state}, pending_kv_bytes and deferred{reason}.\n  Drop the credit and ready wait histograms, the oversized counter and\n  their timestamps.\n- The controller no longer takes metrics or walks its pending entries on\n  every event. It exposes pending, pending_bytes and per-limit deferral\n  counts; collect_metrics reads them once per snapshot and the exporter\n  emits them per DP rank, like the scheduler queue gauges.\n- The deferral marker stays in the controller, keyed by request id.\n\n* refactor(pd): address review on the admission metrics snapshot\n\n- The controller reports pending requests by state (pending_states), so\n  the cancelled-before-ready ordering stays with it; collect_metrics only\n  reads it.\n- PrefillScheduler declares pd_admission = None like Scheduler, and\n  collect_metrics reads the attribute directly.\n- A repeated deferral no longer allocates a throwaway set, and cancel\n  computes its key once.\n- The snapshot test goes through LLMEngine.get_metrics_statistics and the\n  exporter, so it checks the real engine_role label, and follows a request\n  from receiving to ready to its first forward, and another through\n  cancellation until its writes drain.\n\n* docs(pd): count admission deferrals once per admission wait\n\nThe marker is dropped when a request is admitted, cancelled or released,\nso the counter counts a request once per wait, not once per request.\nToday a request waits for admission at most once: admission clears the\nconnector's do_remote_prefill, so a later pass, including a preempted\nrequest's replay, never asks for admission again. Say what holds\nregardless.\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-11-16T12:28:56+08:00",
          "tree_id": "76accd0933d16e0dec80af1d75e0bd033bc89772",
          "url": "https://github.com/ZhiweiYan-96/ATOM/commit/d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9"
        },
        "date": 1794814986409,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2366.55,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425979 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425979 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425979 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425979 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425979 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7413.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334432 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334432 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334432 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334432 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334432 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3952.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=711381 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=711381 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=711381 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=711381 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=711381 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5922.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1066034 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1066034 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1066034 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1066034 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1066034 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7062.45,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271241 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271241 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.9,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271241 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271241 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271241 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2368.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426292 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426292 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426292 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426292 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426292 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7372.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327102 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327102 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327102 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327102 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327102 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3931.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=707602 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=707602 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=707602 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=707602 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=707602 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5952.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1071530 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1071530 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1071530 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1071530 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1071530 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7065.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1271724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2347.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7341.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1321500 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1321500 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.61,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1321500 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1321500 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1321500 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3922.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706060 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706060 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706060 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706060 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706060 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5909.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063663 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063663 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063663 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063663 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063663 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7026.9,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1264842 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1264842 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1264842 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1264842 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1264842 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43817582665"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Mengqing Cao",
            "username": "MengqingCao",
            "email": "cmq0113@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9",
          "message": "feat(pd): export decode admission metrics (#3173)\n\n* feat(pd): export decode admission and deferred-free KV metrics\n\nDecode admission state, per D rank (`atom:pd_admission_*`):\n\n- `pending_requests{state=\"receiving|ready|cancel_draining\"}` and\n  `pending_kv_bytes`: the credits held and the receive-target bytes behind\n  them.\n- `deferred_total{reason}`: requests deferred by a limit, once per request\n  and limit however many passes they wait.\n- `credit_wait_seconds`: first deferral to admission; 0 when admitted at once.\n- `oversized_total` and `ready_wait_seconds`: isolated admissions above the\n  budget, and receive completion to the first local forward.\n\nEvery scheduler also exports `atom:requests_deferred_free` and\n`atom:kv_cache_blocks_deferred_free` (physical blocks, shared prefixes\ncounted once): on P/D prefill, the source KV decode has not pulled yet. They\nare computed with the metrics snapshot, not per step.\n\n* fix(pd): address review on decode admission metrics\n\n- Count kv_blocks_deferred_free with a boolean mask over the pool instead\n  of a Python set of every block id. The snapshot runs on the engine thread\n  every push interval; the set cost 80-2500 ms at P/D backpressure.\n- Keep only the Prometheus counters for admission deferrals and oversized\n  admissions. The parallel `deferred` / `oversized_admissions` fields had no\n  reader outside tests, so the counters were never checked.\n- Register the admission counters without the `_total` suffix in the family\n  name (prometheus_client adds it on export); docs follow.\n- Describe deferred-free as whole-request holds; early-release offload leases\n  return once the source read finishes and are omitted.\n- Test the oversized path, cancel_draining, and that Scheduler passes its\n  metrics to admission.\n\n* test(pd): count deferred-free blocks from a real block table\n\nThe snapshot reads array('i') tables through np.frombuffer. The existing\ncase passed plain lists, so that branch never ran.\n\n* test(pd): a rejected deferred request leaves no admission marker\n\nThe deferral marker now lives in the controller keyed by request id, so a\nrequest rejected from waiting before admission must not leave it behind.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* refactor(pd): drop the P-side deferred-free metrics\n\nrequests_deferred_free and kv_cache_blocks_deferred_free count prefill-side\nKV that decode has not pulled. They do not depend on decode admission, and\nnothing in this PR needs them.\n\n* refactor(pd): export decode admission state from the metrics snapshot\n\n- Keep pending_requests{state}, pending_kv_bytes and deferred{reason}.\n  Drop the credit and ready wait histograms, the oversized counter and\n  their timestamps.\n- The controller no longer takes metrics or walks its pending entries on\n  every event. It exposes pending, pending_bytes and per-limit deferral\n  counts; collect_metrics reads them once per snapshot and the exporter\n  emits them per DP rank, like the scheduler queue gauges.\n- The deferral marker stays in the controller, keyed by request id.\n\n* refactor(pd): address review on the admission metrics snapshot\n\n- The controller reports pending requests by state (pending_states), so\n  the cancelled-before-ready ordering stays with it; collect_metrics only\n  reads it.\n- PrefillScheduler declares pd_admission = None like Scheduler, and\n  collect_metrics reads the attribute directly.\n- A repeated deferral no longer allocates a throwaway set, and cancel\n  computes its key once.\n- The snapshot test goes through LLMEngine.get_metrics_statistics and the\n  exporter, so it checks the real engine_role label, and follows a request\n  from receiving to ready to its first forward, and another through\n  cancellation until its writes drain.\n\n* docs(pd): count admission deferrals once per admission wait\n\nThe marker is dropped when a request is admitted, cancelled or released,\nso the counter counts a request once per wait, not once per request.\nToday a request waits for admission at most once: admission clears the\nconnector's do_remote_prefill, so a later pass, including a preempted\nrequest's replay, never asks for admission again. Say what holds\nregardless.\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-11-16T04:28:56Z",
          "url": "https://github.com/ZhiweiYan-96/ATOM/commit/d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9"
        },
        "date": 1794884447204,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2307.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7300.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1314002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1314002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.62,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1314002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1314002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1314002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3841.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691464 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691464 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691464 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691464 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691464 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5832,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049760 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049760 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049760 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049760 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049760 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6951.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1251350 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1251350 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1251350 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1251350 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1251350 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2314.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416614 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416614 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416614 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416614 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416614 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7272.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309004 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309004 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309004 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309004 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309004 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3838,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690840 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690840 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690840 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690840 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690840 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5818.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047266 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047266 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047266 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047266 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047266 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6981.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1256658 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1256658 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1256658 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1256658 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1256658 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2305.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415030 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415030 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415030 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415030 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415030 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7275.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309627 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309627 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.64,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309627 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309627 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1309627 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3842.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691604 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691604 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691604 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691604 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691604 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5824.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1048386 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1048386 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1048386 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1048386 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1048386 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6949.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250828 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250828 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250828 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250828 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250828 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/43998420059"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Mengqing Cao",
            "username": "MengqingCao",
            "email": "cmq0113@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9",
          "message": "feat(pd): export decode admission metrics (#3173)\n\n* feat(pd): export decode admission and deferred-free KV metrics\n\nDecode admission state, per D rank (`atom:pd_admission_*`):\n\n- `pending_requests{state=\"receiving|ready|cancel_draining\"}` and\n  `pending_kv_bytes`: the credits held and the receive-target bytes behind\n  them.\n- `deferred_total{reason}`: requests deferred by a limit, once per request\n  and limit however many passes they wait.\n- `credit_wait_seconds`: first deferral to admission; 0 when admitted at once.\n- `oversized_total` and `ready_wait_seconds`: isolated admissions above the\n  budget, and receive completion to the first local forward.\n\nEvery scheduler also exports `atom:requests_deferred_free` and\n`atom:kv_cache_blocks_deferred_free` (physical blocks, shared prefixes\ncounted once): on P/D prefill, the source KV decode has not pulled yet. They\nare computed with the metrics snapshot, not per step.\n\n* fix(pd): address review on decode admission metrics\n\n- Count kv_blocks_deferred_free with a boolean mask over the pool instead\n  of a Python set of every block id. The snapshot runs on the engine thread\n  every push interval; the set cost 80-2500 ms at P/D backpressure.\n- Keep only the Prometheus counters for admission deferrals and oversized\n  admissions. The parallel `deferred` / `oversized_admissions` fields had no\n  reader outside tests, so the counters were never checked.\n- Register the admission counters without the `_total` suffix in the family\n  name (prometheus_client adds it on export); docs follow.\n- Describe deferred-free as whole-request holds; early-release offload leases\n  return once the source read finishes and are omitted.\n- Test the oversized path, cancel_draining, and that Scheduler passes its\n  metrics to admission.\n\n* test(pd): count deferred-free blocks from a real block table\n\nThe snapshot reads array('i') tables through np.frombuffer. The existing\ncase passed plain lists, so that branch never ran.\n\n* test(pd): a rejected deferred request leaves no admission marker\n\nThe deferral marker now lives in the controller keyed by request id, so a\nrequest rejected from waiting before admission must not leave it behind.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* refactor(pd): drop the P-side deferred-free metrics\n\nrequests_deferred_free and kv_cache_blocks_deferred_free count prefill-side\nKV that decode has not pulled. They do not depend on decode admission, and\nnothing in this PR needs them.\n\n* refactor(pd): export decode admission state from the metrics snapshot\n\n- Keep pending_requests{state}, pending_kv_bytes and deferred{reason}.\n  Drop the credit and ready wait histograms, the oversized counter and\n  their timestamps.\n- The controller no longer takes metrics or walks its pending entries on\n  every event. It exposes pending, pending_bytes and per-limit deferral\n  counts; collect_metrics reads them once per snapshot and the exporter\n  emits them per DP rank, like the scheduler queue gauges.\n- The deferral marker stays in the controller, keyed by request id.\n\n* refactor(pd): address review on the admission metrics snapshot\n\n- The controller reports pending requests by state (pending_states), so\n  the cancelled-before-ready ordering stays with it; collect_metrics only\n  reads it.\n- PrefillScheduler declares pd_admission = None like Scheduler, and\n  collect_metrics reads the attribute directly.\n- A repeated deferral no longer allocates a throwaway set, and cancel\n  computes its key once.\n- The snapshot test goes through LLMEngine.get_metrics_statistics and the\n  exporter, so it checks the real engine_role label, and follows a request\n  from receiving to ready to its first forward, and another through\n  cancellation until its writes drain.\n\n* docs(pd): count admission deferrals once per admission wait\n\nThe marker is dropped when a request is admitted, cancelled or released,\nso the counter counts a request once per wait, not once per request.\nToday a request waits for admission at most once: admission clears the\nconnector's do_remote_prefill, so a later pass, including a preempted\nrequest's replay, never asks for admission again. Say what holds\nregardless.\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-11-16T04:28:56Z",
          "url": "https://github.com/ZhiweiYan-96/ATOM/commit/d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9"
        },
        "date": 1794967492513,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2313.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416420 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416420 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416420 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416420 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416420 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 6823.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1228272 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.3,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1228272 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1228272 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1228272 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1228272 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3862.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695306 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695306 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695306 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695306 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695306 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5766.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037882 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037882 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037882 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037882 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037882 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6629.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193392 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193392 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193392 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193392 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193392 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2308.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415450 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415450 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415450 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415450 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415450 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7190.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1294288 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1294288 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1294288 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1294288 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1294288 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3860.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694874 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694874 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694874 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694874 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694874 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5758.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036506 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036506 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036506 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036506 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036506 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6902.45,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242441 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242441 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.95,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242441 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242441 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242441 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2321.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417927 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417927 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417927 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417927 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417927 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6840.15,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1231227 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.29,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1231227 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1231227 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1231227 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1231227 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3842.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691630 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691630 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691630 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691630 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691630 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5810.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045895 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045895 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045895 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045895 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045895 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6779.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1220302 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1220302 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1220302 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1220302 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1220302 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44211279576"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Mengqing Cao",
            "username": "MengqingCao",
            "email": "cmq0113@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9",
          "message": "feat(pd): export decode admission metrics (#3173)\n\n* feat(pd): export decode admission and deferred-free KV metrics\n\nDecode admission state, per D rank (`atom:pd_admission_*`):\n\n- `pending_requests{state=\"receiving|ready|cancel_draining\"}` and\n  `pending_kv_bytes`: the credits held and the receive-target bytes behind\n  them.\n- `deferred_total{reason}`: requests deferred by a limit, once per request\n  and limit however many passes they wait.\n- `credit_wait_seconds`: first deferral to admission; 0 when admitted at once.\n- `oversized_total` and `ready_wait_seconds`: isolated admissions above the\n  budget, and receive completion to the first local forward.\n\nEvery scheduler also exports `atom:requests_deferred_free` and\n`atom:kv_cache_blocks_deferred_free` (physical blocks, shared prefixes\ncounted once): on P/D prefill, the source KV decode has not pulled yet. They\nare computed with the metrics snapshot, not per step.\n\n* fix(pd): address review on decode admission metrics\n\n- Count kv_blocks_deferred_free with a boolean mask over the pool instead\n  of a Python set of every block id. The snapshot runs on the engine thread\n  every push interval; the set cost 80-2500 ms at P/D backpressure.\n- Keep only the Prometheus counters for admission deferrals and oversized\n  admissions. The parallel `deferred` / `oversized_admissions` fields had no\n  reader outside tests, so the counters were never checked.\n- Register the admission counters without the `_total` suffix in the family\n  name (prometheus_client adds it on export); docs follow.\n- Describe deferred-free as whole-request holds; early-release offload leases\n  return once the source read finishes and are omitted.\n- Test the oversized path, cancel_draining, and that Scheduler passes its\n  metrics to admission.\n\n* test(pd): count deferred-free blocks from a real block table\n\nThe snapshot reads array('i') tables through np.frombuffer. The existing\ncase passed plain lists, so that branch never ran.\n\n* test(pd): a rejected deferred request leaves no admission marker\n\nThe deferral marker now lives in the controller keyed by request id, so a\nrequest rejected from waiting before admission must not leave it behind.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* refactor(pd): drop the P-side deferred-free metrics\n\nrequests_deferred_free and kv_cache_blocks_deferred_free count prefill-side\nKV that decode has not pulled. They do not depend on decode admission, and\nnothing in this PR needs them.\n\n* refactor(pd): export decode admission state from the metrics snapshot\n\n- Keep pending_requests{state}, pending_kv_bytes and deferred{reason}.\n  Drop the credit and ready wait histograms, the oversized counter and\n  their timestamps.\n- The controller no longer takes metrics or walks its pending entries on\n  every event. It exposes pending, pending_bytes and per-limit deferral\n  counts; collect_metrics reads them once per snapshot and the exporter\n  emits them per DP rank, like the scheduler queue gauges.\n- The deferral marker stays in the controller, keyed by request id.\n\n* refactor(pd): address review on the admission metrics snapshot\n\n- The controller reports pending requests by state (pending_states), so\n  the cancelled-before-ready ordering stays with it; collect_metrics only\n  reads it.\n- PrefillScheduler declares pd_admission = None like Scheduler, and\n  collect_metrics reads the attribute directly.\n- A repeated deferral no longer allocates a throwaway set, and cancel\n  computes its key once.\n- The snapshot test goes through LLMEngine.get_metrics_statistics and the\n  exporter, so it checks the real engine_role label, and follows a request\n  from receiving to ready to its first forward, and another through\n  cancellation until its writes drain.\n\n* docs(pd): count admission deferrals once per admission wait\n\nThe marker is dropped when a request is admitted, cancelled or released,\nso the counter counts a request once per wait, not once per request.\nToday a request waits for admission at most once: admission clears the\nconnector's do_remote_prefill, so a later pass, including a preempted\nrequest's replay, never asks for admission again. Say what holds\nregardless.\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-11-16T04:28:56Z",
          "url": "https://github.com/ZhiweiYan-96/ATOM/commit/d2d5a6bbb2bdc8e7b5ea65cd130f5a010aa373f9"
        },
        "date": 1795051372323,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2377.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=427974 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=427974 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=427974 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=427974 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=427974 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7524.48,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354406 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354406 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354406 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354406 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354406 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3976.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=715848 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=715848 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=715848 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=715848 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=715848 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 6052.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1089461 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1089461 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1089461 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1089461 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1089461 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7192.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1294600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1294600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1294600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1294600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1294600 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2365.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425824 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425824 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425824 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425824 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425824 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7552.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359484 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359484 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359484 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359484 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359484 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3966.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=714053 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=714053 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=714053 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=714053 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=714053 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 6019.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1083515 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1083515 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.01,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1083515 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1083515 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1083515 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7188.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1293906 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1293906 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1293906 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1293906 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1293906 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2367.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426098 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426098 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426098 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426098 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=426098 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6856.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234109 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.29,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234109 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234109 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234109 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234109 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3964.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=713551 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=713551 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=713551 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=713551 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=713551 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5859.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054688 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054688 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054688 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054688 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054688 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6748.99,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1214819 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1214819 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.98,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1214819 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1214819 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1214819 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44426696465"
          }
        ]
      }
    ]
  }
}