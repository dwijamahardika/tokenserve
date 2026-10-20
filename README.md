# tokenserve

An **LLM inference server** in Python/asyncio, built around the two components that decide how a vLLM/TGI-style server behaves under load: an **iteration-level (continuous-batching) scheduler** and a **paged KV-cache manager with refcounted prefix caching**. It speaks the OpenAI completions and chat API with SSE streaming, cancels work when clients go away, pushes back with `429` when full, and exports the latency metrics that matter (TTFT, TPOT). There is no GPU here, so model execution sits behind a small interface and the default backend is a **deterministic simulator with an explicit latency model**. The scheduler, memory manager, API, metrics and load generator are all real.

```
client ──HTTP/SSE──► API (FastAPI)                         admin listener, 127.0.0.1:9090
                       │ validate · auth · 429 · timeouts   /metrics /readyz /debug/state
                       │ disconnect / timeout ──► abort
                       ▼
                 engine loop (one asyncio task, one step at a time)
                       │ schedule ──► Scheduler ─────────► KVCacheManager
                       │              waiting / running     fixed-size blocks, block tables
                       │              token budget          prefix hash → block (refcount, LRU)
                       │              preempt youngest      free + used == total
                       │ execute  ──► Backend:  Simulated (a + b·tokens)  |  HF transformers (CPU)
                       │ update   ──► append tokens · register full blocks · finish
                       ▼
                 incremental detokenize ─► stop strings ─► per-request stream ─► SSE
```

```bash
make run                                   # API on 127.0.0.1:8000, admin on 127.0.0.1:9090
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}'
curl -s localhost:9090/debug/state         # running/waiting sequences, KV blocks, cache hits
curl -s localhost:9090/metrics | grep ^tokenserve_
```

## What it demonstrates

| Concern | Solution | Where |
|---|---|---|
| Continuous batching | Iteration-level scheduling: every step builds a batch from running decodes, chunked prefills and new arrivals under `max_num_batched_tokens` and `max_num_seqs`. Sequences join between decode steps and leave as soon as they finish. Decodes are scheduled before prefill chunks, so a long prompt cannot freeze streams that are already generating. | `scheduler.py` |
| Static baseline | Request-level batching on the *same* engine and KV manager: a batch forms only when the previous one has drained. Deliberately generous: finished members cost nothing (no padding), and it gets paging, chunked prefill and prefix caching too. | `StaticScheduler` |
| Paged KV cache | Fixed-size blocks and per-sequence block tables. Blocks are allocated on prefill and on decode growth, all-or-nothing. A watermark is reserved at admission so running sequences can grow. The backend receives block tables plus a slot mapping, which is what a paged-attention kernel consumes. | `kv_cache.py` |
| Preemption | By **recompute**: when a step cannot get blocks, the youngest running sequence loses its KV, is re-queued in arrival order and later recomputes prompt+output (partly from the prefix cache, since its freed blocks stay cached). Tests check that clients see byte-identical output. | `Scheduler._preempt` |
| Prefix caching | Each full block is registered under a chained SHA-256 of (parent hash, tokens). Shared blocks are refcounted. Free-but-cached blocks are LRU-evicted only after plain free blocks run out. Sequences release blocks tail-first so prefixes outlive suffixes. At least one prompt token is always recomputed, to get logits. | `kv_cache.py` |
| No starvation | `waiting` and `running` are both kept in arrival order. Admission is strict FCFS with head-of-line blocking, and preemption takes the youngest, so the oldest request always progresses (argument in the scheduler docstring, plus a randomised test). | `scheduler.py` |
| OpenAI API | `POST /v1/completions` and `/v1/chat/completions`, both streaming (SSE, `stream_options.include_usage`) and non-streaming. Stop sequences are held back so a partial match never streams. Errors are OpenAI-shaped (400/401/404/413/429/503/504), `max_tokens` is capped, and optional bearer auth uses a constant-time compare. | `api/app.py` |
| Cancellation | A client disconnect (streaming or not) or a per-request timeout calls `engine.abort`, and the blocks are freed at the next step boundary. Tested via raw ASGI and over a real TCP socket. | `api/app.py`, `engine.py` |
| Backpressure | The waiting queue is bounded by count and by queued prompt tokens (default: one KV cache's worth). Overflow gets `429` with `Retry-After` taken from an EWMA of queue time. Requests that can *never* fit (context length or KV capacity) get `400` at the door. | `Engine.add_request` |
| Metrics | TTFT, TPOT, inter-token latency, queue time, end-to-end latency, batch size (sequences and tokens), running/waiting, KV usage, prefix-cache hits/queries and hit rate, preemptions, rejections. Prometheus on a separate admin listener bound to loopback by default. | `metrics.py`, `api/admin.py` |
| Backends | `SimulatedBackend`: step latency `a + b·tokens`, deterministic tokens, timer-error compensation, and an optional **shadow KV** that verifies every block table. `HFBackend`: `transformers` on CPU (optional extra). | `backends/` |
| Load generator | Open-loop Poisson arrivals, prompt/output length distributions, shared-prefix workloads. Latency is measured from the *scheduled* arrival time (no coordinated omission). Reports p50/p90/p99 TTFT and TPOT. | `bench/` |
| Testing | A Hypothesis state machine for the KV manager, an end-to-end engine property test (random workloads and aborts on a tiny cache, with the shadow-KV verifier on), scheduler/API/metrics/CLI tests, and real-socket disconnect tests. | `tests/` |

Runtime dependencies: `fastapi`, `uvicorn`, `pydantic`, `prometheus-client` (plus optional `uvloop`/`httptools`). No inference framework.

## Run

```bash
make venv                      # .venv with dev + speed extras
make run                       # simulated backend, continuous batching, prefix caching on
.venv/bin/tokenserve --help    # every option; each is also TOKENSERVE_<FLAG> in the env
```

Common knobs: `--scheduler continuous|static`, `--num-blocks` / `--block-size` (KV capacity), `--max-num-seqs`, `--max-num-batched-tokens` (per-step token budget), `--[no-]chunked-prefill`, `--[no-]prefix-caching`, `--max-waiting-seqs` / `--max-waiting-tokens` (backpressure), `--request-timeout`, and `--sim-base-latency-ms` / `--sim-per-token-latency-ms` (the simulator's `a` and `b`).

Auth is off unless `TOKENSERVE_API_KEY` is set. It is read from the environment only, never from a flag, because flags end up in shell history and `ps`. See `.env.example`. Both listeners bind `127.0.0.1` unless told otherwise. The Docker image binds `0.0.0.0` because container loopback is unreachable from the host:

```bash
docker build -t tokenserve:dev .
docker run --rm -p 8000:8000 -p 127.0.0.1:9090:9090 tokenserve:dev
```

**Real model (optional):** `pip install -e '.[hf]'` with a CPU build of torch (`--index-url https://download.pytorch.org/whl/cpu`), then `tokenserve --backend hf --hf-model <id-or-path>`. See [Status](#status).

## Design notes

**Why continuous batching wins.** A decode step costs roughly the same whether it carries 1 sequence or 30, because it is dominated by streaming the weights from memory (the `a` term). Batching is the only way to amortise that cost. Static batching does batch, but at request granularity: a batch lives as long as its *longest* member, so slots of finished sequences sit idle, and a new request waits for the whole batch to drain before it gets its first token. With output lengths between 16 and 512 tokens, that wait is several seconds even before the server saturates (table below: 3.5 s median TTFT for static versus 61 ms for continuous at the same 4 req/s). Iteration-level scheduling refills a freed slot on the very next step.

**Why KV memory, not FLOPs, bounds the batch.** Back-of-envelope for a Llama-2-7B-shaped model in fp16: KV per token = 2 (K,V) × 32 layers × 32 heads × 128 dims × 2 bytes = **512 KiB**, so a 4k-token sequence holds 2 GiB of cache. On an 80 GB GPU with ~13.5 GB of weights, ~60 GB is left for KV: about 120k tokens, or 30 sequences at full length. A decode step for those 30 sequences needs ~0.4 TFLOP (≈ 2 × params × batch), about 1.3 ms at A100 peak. Reading the weights alone takes ~7 ms, and reading their KV takes longer still. Compute sits mostly idle and memory runs out first. That is why the scheduler's admission and preemption decisions are phrased in KV blocks, and why paging matters: reserving `max_model_len` slots per request (what pre-paging servers did) wastes most of that memory when typical sequences are much shorter. These are published model shapes and spec-sheet figures, not measurements from this repo. The repo's own measured version of the point is the *KV cache / 4* benchmark row: same cost model, same code, a quarter of the memory, and the batch shrinks from 34 to 15 sequences.

**Preemption: recompute vs swap, and who to evict.** Recompute throws away the victim's KV and re-runs prompt+output later. That costs prefill compute, which is the cheap kind (dense, compute-bound), and blocks that were freed but not yet evicted come back as prefix-cache hits. Swap would copy the KV to host memory (1 GiB for a 2k-token sequence of the model above, tens of ms over PCIe each way) and needs a CPU block pool with its own accounting. Recompute is simpler, and it is vLLM's default for single-sequence requests. Swap is not implemented. The victim is the youngest running sequence: it has the least work invested, and evicting youngest-first is what keeps FCFS fair. A step that had to preempt admits nobody new, otherwise the newcomer would immediately be the next victim.

**Chunked prefill and decode-first ordering.** Under this cost model, a 1,000-token prompt scheduled in one step makes that step about 10× longer than a decode step, and every running stream stalls for it. With chunking, the token budget caps any single step. Decodes are scheduled first and prefill chunks fill what is left, so the cost of a new arrival is spread over a few steps instead of hitting everyone's inter-token latency at once. The budget is a trade-off. In the benchmark it is 2,048 tokens, so a ~1,100-token prefill lands in a single step, and with prefix caching off TPOT p50 is 54 ms versus 13 ms with it on. A smaller budget would cap that stall at the cost of TTFT; this repo does not measure that trade-off.

**Prefix-cache details that matter.** (1) Hashes chain through the parent block, so one hash identifies a whole prefix and lookup is a walk until the first miss. (2) SHA-256, not `hash()`: with a 64-bit non-cryptographic hash, a crafted prompt could collide with another tenant's prefix and be served the wrong KV. (3) At least one token is always recomputed, because the admitting step needs logits to sample from. (4) Two sequences that compute the same block concurrently both keep private copies, and only the first is registered. (5) Blocks that hold nothing reusable are handed out before any cached block is evicted.

**Aborts happen between steps.** `abort()` only records the request id. The engine applies aborts at the next step boundary, so a backend never sees its blocks freed (and possibly reallocated) while a step that uses them is in flight. The same rule keeps all scheduler and KV mutation on one task, so there are no locks.

**A real measurement bug, found and fixed.** Event-loop timers are only accurate to about 1 ms (stdlib asyncio overshot a 10.3 ms sleep by ~1 ms on average, and uvloop rounds to whole milliseconds). For a simulator whose step is ~10 ms, that is a 5-10% systematic error that would have leaked straight into every benchmark number. `SimulatedBackend` now carries each sleep's error into the next one, bounded at ±5 ms, and a test pins total simulated time to the model. The benchmark also reports *step overhead*, which is engine wall time per step relative to `a + b·tokens`. It covers the Python scheduling, bookkeeping and streaming work that is not hidden behind simulated GPU time, and it came out at +1.3% to +4.4%.

**The shadow KV verifier.** With `verify_kv=True`, the simulated backend writes each computed token id into a fake KV slot through the block table and slot mapping, then reads every sequence's full context back through its table. A shared block that gets overwritten, a cached block evicted while still referenced, or a wrong table after a prefix hit fails loudly instead of quietly producing wrong text. The engine property test runs with it on.

**Not included (scope):** swap-based preemption, priority classes and per-tenant fairness, speculative decoding, multi-LoRA, `n > 1` / beam search, logprobs, tensor parallelism. None of these would change the architecture.

## Benchmarks

Measured with `make bench` on the **simulated backend, with a = 10 ms per step and b = 0.1 ms per batched token. This measures the scheduler and the server, not GPU kernels.** Machine: a 4 vCPU cloud VM (shared with other workloads during the run), Python 3.13 with uvloop. Server and load generator ran as separate processes on that machine.

Server config: 2,048 KV blocks × 16 tokens (32k tokens), `max_num_seqs` 64, `max_num_batched_tokens` 2,048, chunked prefill on. Queue limits are raised so every request is served and overload shows up as latency rather than `429`s. Each run offers 30 s of open-loop Poisson arrivals and then waits for every request to finish. All 1,560 requests completed.

- **mixed:** random prompts of 32-512 tokens, outputs of 16-512 tokens (`ignore_eos`), no shared prefixes.
- **shared-prefix:** each prompt is one of 4 random 1,024-token "system prompts" plus a unique 32-128-token suffix. Outputs are 16-128 tokens.

| Scenario | Workload | req/s | out tok/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 / p90 / p99 (ms) | mean batch | preempt | prefix hit |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| static batching | mixed @ 4 req/s | 3.02 | 790 | 3,501 / 5,861 / 7,373 | 15.5 / 19.7 / 33.0 | 10.6 | 0 | n/a |
| continuous batching | mixed @ 4 req/s | 3.42 | 895 | 61.2 / 80.2 / 96.6 | 15.2 / 16.4 / 17.8 | 12.6 | 0 | n/a |
| static batching | mixed @ 8 req/s | 5.21 | 1,315 | 6,667 / 10,839 / 11,745 | 20.0 / 31.5 / 51.8 | 22.0 | 0 | n/a |
| continuous batching | mixed @ 8 req/s | 6.63 | 1,674 | 75.0 / 124 / 201 | 24.5 / 26.7 / 31.2 | 33.9 | 0 | n/a |
| continuous, KV cache / 4 | mixed @ 8 req/s | 3.94 | 994 | 12,731 / 23,977 / 26,261 | 16.2 / 20.2 / 33.1 | 15.5 | 143 | n/a |
| continuous, prefix cache off | shared-prefix @ 8 req/s | 6.38 | 482 | 4,095 / 5,726 / 6,086 | 54.2 / 61.7 / 71.3 | 24.3 | 0 | - |
| continuous, prefix cache on | shared-prefix @ 8 req/s | 7.49 | 566 | 34.1 / 47.2 / 136 | 12.9 / 14.6 / 15.6 | 7.4 | 0 | 91% |

How to read the columns:
- *req/s* is completed requests divided by the time from first arrival to last completion. It includes the drain tail, so it only approaches the offered rate when the server keeps up.
- *TTFT* is measured by the client from each request's scheduled arrival time.
- *TPOT* is (last token − first token) / (tokens − 1), per request.
- *mean batch* is sequences per engine step, from the server's histogram.
- *preempt* counts recompute preemptions.

What the numbers say:

- **Static vs continuous at 4 req/s.** Here the difference is mostly latency. Static's median TTFT is **3.5 s versus 61 ms**, because every new request waits for the running batch's longest member, and median end-to-end latency is 7.4 s versus 3.9 s. Static is already near its capacity at this load: its longer drain tail is most of the req/s gap.
- **At 8 req/s, static falls behind.** It serves 5.2 req/s against 8 offered, so its queue grows throughout the arrival window and TTFT p99 reaches 11.7 s. Continuous keeps up (TTFT p99 201 ms) with a mean batch of 34 versus 22. Its TPOT rises from 15 to 25 ms because each step carries more tokens, which is the `b·tokens` price of a bigger batch.
- **A quarter of the KV cache** (512 blocks, 8k tokens) under the same 8 req/s load: only ~15 sequences stay resident instead of 34, the scheduler preempts 143 times and recomputes **34% more prefill tokens** than the prompts contained, and the server falls behind (3.9 req/s, TTFT p50 12.7 s). Same compute model and same code with a quarter of the memory gives less than half the batch.
- **Prefix cache.** 91% of prompt tokens are served from cache, and the prefill tokens actually computed drop from 264k to 23k. TTFT p50 goes from 4.1 s to 34 ms, and the server moves from falling behind to keeping up. TPOT improves too (54 to 13 ms), because steps are no longer full of 1,100-token prefills.
- **Overhead.** Engine wall time per step exceeded `a + b·tokens` by **1.3-4.4%** across the runs. That is the Python scheduling, bookkeeping and streaming cost that is not hidden behind simulated device time.

`make bench` takes about 5 minutes. It writes `bench/results/latest.md` and a JSON file with every percentile and the server-side counters (prefill tokens computed, mean tokens per step, step overhead). Absolute numbers will move with the machine. The shape of the comparison should not.

## Layout

```
src/tokenserve/
  sequence.py         sequence state: prompt/output tokens, computed-token count, timestamps
  kv_cache.py         paged block manager + prefix cache (hash chain, refcounts, LRU)
  scheduler.py        continuous and static schedulers, preemption, update after a step
  engine.py           async step loop, request streams, aborts, backpressure, metrics hooks
  tokenizer.py        byte tokenizer, incremental detokenizer, stop-string matcher
  metrics.py          Prometheus registry
  backends/
    base.py           Backend interface, BatchItem (block table, slot mapping)
    simulated.py      latency model, deterministic tokens, shadow-KV verifier
    hf.py             transformers on CPU (optional extra)
  api/
    app.py            OpenAI-compatible routes, SSE, cancellation, errors, auth
    admin.py          /metrics /healthz /readyz /debug/state
  server.py, cli.py   process wiring: two uvicorn listeners on one loop, flags + env
bench/
  loadgen.py          open-loop Poisson load generator (also a CLI)
  run.py              the benchmark matrix behind `make bench`
tests/                unit, property-based, API and end-to-end tests
```

## Develop

```bash
make test          # full suite, ~5 s
make test-props    # property tests with ~5x more examples (CI profile)
make lint          # ruff check + ruff format --check + mypy --strict
make fmt
make bench         # the matrix above, ~5 min; writes bench/results/
python -m bench.loadgen --url http://127.0.0.1:8000 --qps 8 --num-requests 200 \
    --prompt-len uniform:32:512 --output-len uniform:16:512
```

## Status

**Real and tested:** the continuous and static schedulers, the paged KV-cache manager with prefix caching and recompute preemption, the OpenAI-compatible API with streaming, cancellation, timeouts, backpressure and auth, the Prometheus metrics, and the load generator. 135 tests pass and one (the HF backend test, below) is skipped. `ruff` and `mypy --strict` are clean.

**Simulated:** model execution. The default backend sleeps for `a + b·tokens` and emits deterministic text. The benchmark numbers therefore measure *scheduling policy and server overhead under an assumed cost model*, not kernels and not model quality. The cost model leaves out two things a real GPU has: attention cost that grows with context length (KV reads per decode token, quadratic prefill attention), and the roofline shape of real step times, which behave more like max(memory time, compute time) than a sum.

**Written but not exercised:** `HFBackend` (`transformers` on CPU) and its test, which runs a tiny randomly initialised GPT-2 against a cache-less greedy reference, with and without forced preemption. In the environment this was developed in, the Hugging Face Hub was unreachable and no CPU build of torch was installable, so the test was skipped. Treat the backend as unverified until `pip install -e '.[hf]'` plus `pytest tests/test_hf_backend.py` passes somewhere. Likewise, the Docker image and the Python 3.11 CI leg were not run locally (CI builds and runs them); locally everything ran on Python 3.13.

**What a GPU run would add:**
- A backend that hands the block tables and slot mappings this engine already produces to a real paged-attention kernel (vLLM's or FlashInfer's).
- Sizing `num_blocks` from free device memory after a profiling forward pass, instead of a flag.
- Fitting `a` and `b` (plus a context-length term) from measured step times, so the simulator can be checked against hardware.
- CUDA graphs for small decode batches.
- Swap-to-host preemption, compared against recompute.
- Overlapping CPU scheduling with device execution. The measured Python step overhead here is a few percent of a 10-25 ms simulated step; against real decode steps of a few milliseconds it would not be, which is why production engines move scheduling off the critical path.
