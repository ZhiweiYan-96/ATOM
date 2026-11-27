window.BENCHMARK_DATA = {
  "lastUpdate": 1795747692021,
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
        "date": 1795139108384,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2366.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425942 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425942 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425942 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425942 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=425942 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7433.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338017 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338017 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338017 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338017 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338017 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3927.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706999 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706999 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706999 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706999 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=706999 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5902.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1062390 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1062390 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1062390 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1062390 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1062390 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7104.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1278782 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1278782 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1278782 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1278782 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1278782 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2383.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=428945 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=428945 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=428945 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=428945 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=428945 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 6951.75,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1251315 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.25,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1251315 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1251315 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1251315 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1251315 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3958.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=712502 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=712502 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=712502 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=712502 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=712502 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5818.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1047360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6461,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162980 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162980 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162980 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162980 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162980 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2304.08,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414734 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414734 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414734 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414734 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414734 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6748.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1214708 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1214708 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1214708 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.27,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1214708 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1214708 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3764.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=677536 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=677536 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=677536 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=677536 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=677536 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5676.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1021780 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1021780 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1021780 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1021780 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1021780 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6600.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.02,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44646548857"
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
        "date": 1795223319392,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3156.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568120 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568120 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568120 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568120 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568120 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10560.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900817 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900817 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900817 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900817 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900817 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5245.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944261 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944261 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944261 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944261 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944261 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8370.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1506623 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1506623 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1506623 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.97,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1506623 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1506623 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 10121.86,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1821935 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1821935 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.32,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1821935 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1821935 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1821935 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3140.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=565330 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=565330 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=565330 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=565330 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=565330 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10559.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900637 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900637 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900637 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.25,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900637 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1900637 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5232.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=941864 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=941864 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=941864 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=941864 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=941864 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8353.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1503712 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1503712 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1503712 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.96,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1503712 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1503712 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10107.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1819362 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1819362 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1819362 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1819362 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1819362 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3135.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=564380 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=564380 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=564380 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=564380 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=564380 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10566.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1902011 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1902011 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1902011 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1902011 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1902011 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5236.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942560 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942560 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942560 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942560 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942560 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8355.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1504052 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1504052 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1504052 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.98,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1504052 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1504052 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 10098.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817738 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817738 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817738 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817738 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817738 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44840106603"
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
        "date": 1795306192469,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3101.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=558324 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=558324 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=558324 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=558324 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=558324 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10479.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1886360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1886360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1886360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1886360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1886360 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5210.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937831 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937831 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937831 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937831 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937831 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8183.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1473112 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1473112 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1473112 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1473112 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1473112 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9717.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1749209 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1749209 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1749209 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1749209 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1749209 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3151.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567188 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567188 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567188 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567188 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567188 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10233.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1841993 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1841993 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1841993 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1841993 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1841993 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5259.15,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=946647 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=946647 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=946647 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=946647 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=946647 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8155.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1468045 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1468045 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1468045 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1468045 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1468045 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 9632.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1733776 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1733776 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1733776 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1733776 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1733776 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3150.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567144 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567144 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567144 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567144 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567144 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 9788.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1761908 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1761908 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1761908 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1761908 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1761908 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5183.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=933013 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=933013 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=933013 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=933013 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=933013 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8228.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1481163 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1481163 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1481163 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.01,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1481163 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1481163 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9529.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1715338 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1715338 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1715338 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.95,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1715338 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1715338 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/44959465597"
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
        "date": 1795394506513,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2319.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417478 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417478 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417478 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417478 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417478 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7172.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1290982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1290982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1290982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1290982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1290982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3823.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688210 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688210 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688210 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688210 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688210 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5764.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1037677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6884.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1239140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1239140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1239140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1239140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1239140 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2321.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417862 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417862 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417862 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417862 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417862 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7241.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1303423 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1303423 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1303423 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1303423 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1303423 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3881.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698677 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5855.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053904 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053904 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053904 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053904 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053904 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6940.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1249320 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1249320 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1249320 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1249320 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1249320 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2334.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420221 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420221 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420221 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420221 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420221 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7177.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1291918 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1291918 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1291918 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1291918 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1291918 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3859.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694724 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5859.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054673 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054673 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054673 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054673 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054673 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6909.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243761 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243761 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243761 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243761 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243761 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45083526198"
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
        "date": 1795496961069,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3166.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10630.4,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1913472 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1913472 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1913472 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1913472 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1913472 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5296.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=953443 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=953443 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=953443 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=953443 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=953443 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8429.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1517394 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1517394 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1517394 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1517394 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1517394 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 10152.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1827411 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1827411 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1827411 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1827411 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1827411 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3170.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570629 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570629 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570629 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570629 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=570629 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10645.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916184 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916184 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916184 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.27,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916184 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916184 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5286.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=951584 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=951584 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=951584 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=951584 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=951584 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8419.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1515556 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1515556 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1515556 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1515556 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1515556 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10133.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1824002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1824002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1824002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1824002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1824002 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3158.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568567 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568567 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568567 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568567 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=568567 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10645.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916117 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916117 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916117 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.26,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916117 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1916117 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5272.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=948962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=948962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=948962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=948962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=948962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8400.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1512048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1512048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1512048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1512048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1512048 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 10197.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1835497 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1835497 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.31,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1835497 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1835497 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1835497 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45322832527"
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
        "date": 1795577189770,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2326.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418844 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418844 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418844 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418844 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418844 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7052.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1269490 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1269490 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1269490 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1269490 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1269490 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3831.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=689592 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=689592 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=689592 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=689592 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=689592 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5733.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1032025 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1032025 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1032025 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1032025 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1032025 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6826.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228702 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.13,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228702 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228702 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228702 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228702 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2316.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416892 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416892 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416892 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416892 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416892 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 6857.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234398 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.29,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234398 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234398 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234398 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1234398 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3837.9,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690822 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690822 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690822 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690822 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690822 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5775.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1039503 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1039503 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1039503 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1039503 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1039503 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6726.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1210785 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1210785 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.99,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1210785 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1210785 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1210785 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2287.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411754 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411754 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411754 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411754 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411754 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6887.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1239784 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.28,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1239784 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1239784 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1239784 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1239784 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3795.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683232 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683232 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683232 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683232 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683232 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5645.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1016264 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1016264 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1016264 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1016264 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1016264 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6657.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198303 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198303 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.01,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198303 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198303 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198303 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45531048517"
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
        "date": 1795661526842,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2307.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=415299 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7249.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304860 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304860 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304860 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304860 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304860 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3858.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694580 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694580 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694580 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694580 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694580 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5789.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042032 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042032 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042032 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042032 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042032 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6824.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228328 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.13,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228328 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228328 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228328 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1228328 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2321.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417830 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417830 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417830 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417830 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417830 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7113.6,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1280448 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1280448 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1280448 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1280448 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1280448 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3820.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687651 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687651 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687651 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687651 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687651 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5798.45,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1043721 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1043721 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1043721 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1043721 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1043721 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6912.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1244258 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1244258 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1244258 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1244258 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1244258 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2323.3,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418194 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418194 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418194 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418194 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418194 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7196.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295416 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295416 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295416 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295416 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295416 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3863.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695378 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695378 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695378 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695378 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=695378 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5784.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1041283 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1041283 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1041283 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1041283 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1041283 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6935.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1248452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1248452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1248452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1248452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1248452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45737634875"
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
        "date": 1795747691598,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2354.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=423771 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=423771 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=423771 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=423771 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=423771 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7324.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1318405 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1318405 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1318405 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1318405 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1318405 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3887.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699727 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699727 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699727 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699727 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699727 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5889.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1060046 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1060046 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1060046 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1060046 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1060046 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6993.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1258856 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1258856 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.92,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1258856 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1258856 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1258856 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2333.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420077 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420077 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420077 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420077 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420077 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7335.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1320452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1320452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1320452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1320452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1320452 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3881.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698748 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698748 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698748 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698748 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=698748 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5857.95,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054431 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054431 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054431 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054431 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1054431 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6983.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1257023 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1257023 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1257023 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1257023 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1257023 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2339.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421123 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421123 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421123 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421123 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421123 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7283.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1310962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1310962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1310962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1310962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1310962 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3891.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=700470 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=700470 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=700470 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=700470 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=700470 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5844.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1051982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1051982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1051982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1051982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1051982 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7035.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1266402 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1266402 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1266402 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1266402 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1266402 Run: https://github.com/ZhiweiYan-96/ATOM/actions/runs/45923695697"
          }
        ]
      }
    ]
  }
}