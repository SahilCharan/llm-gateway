# llm-gateway

[![CI](https://github.com/SahilCharan/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/SahilCharan/llm-gateway/actions/workflows/ci.yml)

A small, provider-neutral LLM gateway. Application code asks for a **job**
(`vision`, `qc`, `prompt`, `research`, `fast`, ...). The gateway sends it down that
job's ordered chain of `provider:model` routes, decides per error whether to
retry, fail over or stop, enforces a whole-call deadline, and writes the USD cost
of every call to a SQLite ledger.

Standard library only (`urllib`, `sqlite3`, `tomllib`). Python 3.11+. No vendor SDKs.

```python
from llm_gateway import Ledger, build_gateway, load_config

gateway = build_gateway(load_config(), ledger=Ledger("llm_gateway_ledger.sqlite3"))
result = gateway.chat("fast", "Summarize this ticket in one line: ...")
print(result.text, result.route, result.usd)
```

## Why a gateway

- **Vendor lock-in.** When call sites name a vendor and a model, switching
  providers means editing every one of them. With a gateway they name a job, and
  switching providers is a config change.
- **Cost.** Different jobs need different models. A cheap model is often good
  enough for classification or QC. You can only prove that when every call is
  priced and attributed to its job.
- **Outages and limits.** Providers rate-limit, return 5xx, stall and run out of
  prepaid credit. A chain of routes keeps the feature working. The failover rules
  also have to be careful: a request that failed because it was malformed should
  not be sent to a second provider.

## Routing

```toml
[jobs]
fast   = ["deepseek:deepseek-chat", "openrouter:openai/gpt-4o-mini"]
vision = ["openrouter:qwen/qwen-2.5-vl-72b-instruct", { route = "gemini:gemini-2.0-flash", timeout_s = 45 }]
```

- Routes are tried in order. The first success wins.
- A route is `provider:model`, split on the first colon, so OpenRouter ids such as
  `openrouter:vendor/model:free` work.
- Jobs without a chain use `default` if it exists. Otherwise the call raises a
  config error that names the env var to set.
- `LLM_ROUTE_<JOB>=a:x,b:y` overrides a job's chain from the environment, so ops
  can reroute without a deploy.
- Built-in providers: `deepseek`, `openrouter`, `openai` (any OpenAI-compatible
  `/chat/completions` host) and `gemini` (native `generateContent`). You can add
  any other OpenAI-compatible host under `[providers.<name>]` with a `base_url`.
  API keys are read from env vars and never from the config file.

See [`examples/llm-gateway.example.toml`](examples/llm-gateway.example.toml) for every option.

## Failure classification

Every provider failure maps to one of three kinds, and the gateway acts on the kind:

| Signal | Kind | What the gateway does |
|---|---|---|
| 429, 408, any 5xx, timeout, connection error, non-JSON body | `retryable` | Waits `backoff_s`, or `Retry-After` if it is no longer than `max_backoff_s`. Retries **once** on the same route, then fails over. A longer `Retry-After` fails over at once. |
| 402, or a body that mentions insufficient balance/quota/credits (including OpenAI-style `429 insufficient_quota`) | `failover` | Goes to the next route right away with no retry. An empty prepaid balance is a reason to use another route, not an outage. |
| 401, 403, 404, or no API key configured | `failover` | Goes to the next route. The route is misconfigured and the request never reached a model. |
| 400, 413, 422 and other 4xx | `permanent` | Raises `PermanentRequestError` with **no failover**. A bad request would fail on the next provider too, and might be billed twice. |

Time limits:

- **Whole-call deadline** (`deadline_s`). One budget covers every retry, backoff
  and failover. When it runs out, the call raises `DeadlineExceeded`.
- **Per-route timeout** (`timeout_s`). Each attempt gets
  `min(route timeout, time left on the deadline)`, so a slow first route cannot
  use up the budget meant for the routes after it.

If every route fails, the call raises `AllRoutesFailed`. Its `attempts` list every
route tried, with the outcome, latency and error.

## Cost ledger

Every attempt, including failures, goes to SQLite with job, route, outcome,
tokens (input / output / cached input), latency, USD and error text. Prices are a
plain table in USD per 1M tokens that you edit:

```toml
[prices."deepseek:deepseek-chat"]
input = 0.27          # placeholders: copy current numbers from the provider
output = 1.10
cached_input = 0.07   # optional; defaults to the input price
```

`usd = (uncached_in * input + cached_in * cached_input + out * output) / 1e6`.
Token counts come from the provider's own `usage` field, including OpenAI-style
`cached_tokens` and DeepSeek's `prompt_cache_hit_tokens`. A successful route with
no price is still logged and marked as unpriced in the report. Failed attempts are
logged at $0.

```console
$ llm-gateway report                 # or --since-hours 24, --ledger path.sqlite3
```

The report shows success/attempt counts and retry, failover and permanent-error
counts per route. That makes it a reliability view as well as a cost view.

## Rollback

```bash
LLM_GATEWAY_LEGACY=1 LLM_GATEWAY_LEGACY_ROUTE=openai:gpt-4o-mini
```

With this setting every job, configured or not, goes to the single legacy route.
Unset it, or set it to `0`, to go back to the job chains. The gateway refuses to
start if the flag is on and no legacy route is set. `llm-gateway routes` prints
the chains that are actually in effect and shows a banner when rollback is active.

## JSON mode

```python
from llm_gateway import chat_json

out = chat_json(gateway, "qc", "Grade this caption...", required_keys=["verdict", "score"])
out.data       # validated dict
out.repaired   # True if the repair re-prompt was needed
```

`chat_json` asks the provider for JSON (`response_format` or `responseMimeType`)
and removes code fences. It rejects:

- empty replies and empty objects
- replies that are not objects
- objects with a blank key
- objects missing a required key

On a rejected reply it sends **one** repair re-prompt that includes the rejected
reply and the reason, then gives up with `JSONValidationError`. The repair call is
a normal gateway call, so its cost shows up in the ledger.

## Quickstart

```bash
git clone https://github.com/SahilCharan/llm-gateway && cd llm-gateway
python -m pip install -e ".[dev]"
python -m pytest                       # no keys, no network (except a 127.0.0.1 stub)
python examples/simulate_traffic.py    # no keys, deterministic
```

Against real providers:

```bash
export DEEPSEEK_API_KEY=...            # and/or OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
export LLM_ROUTE_FAST="deepseek:deepseek-chat,openrouter:openai/gpt-4o-mini"
llm-gateway ask --job fast "Give me three names for a cat"
llm-gateway ask --job fast --json-keys names "Three cat names as {\"names\": [...]}"
llm-gateway routes
llm-gateway report
```

`ask` prints the reply on stdout and a cost line on stderr:
`[route | in+out tok | $usd | ms | attempts]`.

## Sample output

`examples/simulate_traffic.py` sends 400 calls through three fake providers:

- `alpha` is cheap, rate-limited, and runs out of credit after 220 successful calls.
- `beta` is mid-priced, has 5xx errors, and stalls for 12s about 3% of the time
  (past the 8s route timeout).
- `gamma` is expensive and reliable.

About 1% of the requests are malformed. Prices and models are made up. The output
below is copied from a real run:

```text
simulated 400 calls (seed=7)
  served                                     390
  served after failover                      92
  rejected as bad request (not failed over)  9
  all routes failed                          1

job     route               ok/att  retry  failover  perm  in_tok  out_tok       usd    usd/ok  avg_ms
------------------------------------------------------------------------------------------------------
fast    beta:text-mini       79/94     14         0     1   34511     8434  0.013649  0.000173    1373
fast    alpha:text-lite        1/2      0         1     0     155       34  0.000029  0.000029    1121
qc      gamma:text-max       34/35      1         0     0   15904     3289  0.058120  0.001709    2194
qc      alpha:text-lite     70/114     11        32     1   29819     6577  0.005613  0.000080    1011
vision  beta:vision-std      54/67     13         0     0   22982     5610  0.036338  0.000673    1356
vision  alpha:vision-lite  149/225     14        55     7   67402    14738  0.018953  0.000127     968
vision  gamma:vision-max       3/3      0         0     0    1037      441  0.007003  0.002334    2376
------------------------------------------------------------------------------------------------------
TOTAL                      390/540     53        88     9  171810    39123  0.139704  0.000358    1230

rollback check with LLM_GATEWAY_LEGACY=1:
  vision  -> gamma:text-max
  qc      -> gamma:text-max
  fast    -> gamma:text-max
```

How to read it:

- When `alpha` ran out of credit, its 402s show up in the `failover` column.
  Those `vision` calls moved to `beta` and `qc` calls moved to `gamma` without
  retries.
- The 9 malformed requests are in `perm`. Each one was sent once and never failed
  over.
- The one hard failure was a `fast` call. `beta` returned 503 twice, and the
  fallback `alpha` was already out of credit.

## Layout

```text
src/llm_gateway/
  config.py     jobs -> routes, TOML/env loading, rollback flag
  errors.py     ErrorKind + HTTP classification
  providers.py  OpenAI-compatible + Gemini adapters over urllib
  gateway.py    routing loop: retries, failover, deadline, ledger writes
  jsonmode.py   JSON validation + one bounded repair
  ledger.py     price table, SQLite ledger, report formatting
  cli.py        llm-gateway ask | report | routes
  testing.py    FakeProvider, FakeClock (used by tests and the example)
tests/          63 tests: FakeProvider unit tests + a local http.server stub
examples/       traffic simulation + annotated example config
```

## Limitations

- Synchronous only. No streaming, tool calls or embeddings.
- Messages are passed through to OpenAI-compatible hosts as given, so multimodal
  content parts work there. The Gemini adapter is text-only, and it has only been
  tested against a local stub of the documented response shape.
- A request that times out on the client side may still have been processed and
  billed by the provider. The ledger cannot see that cost. Timeouts are retried
  once by design. If you cannot afford that, set `retries_per_route = 0`.
- Prices are whatever you put in the table. The ledger is only as accurate as the
  table.

## Background

Generalized from a vendor-exit migration I led on VirtualPhotography.ai (WorkinX Digital): ~25 LLM call sites moved to job-routed chains with per-call costing and one-flag rollback; measured vision-call cost fell from ~$0.002 to ~$0.00016.

## License

MIT, see [LICENSE](LICENSE).
