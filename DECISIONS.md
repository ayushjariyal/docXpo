# Design Decisions

A running log of *why* docXpo is built the way it is. One section per phase.

---

## Phase 1 — Skeleton

### What I built

An async FastAPI application skeleton with everything the later phases need
already wired in, but no business logic yet:

| Piece | Location |
|---|---|
| Typed configuration | [app/core/config.py](app/core/config.py) |
| Structured logging | [app/core/logging.py](app/core/logging.py) |
| Request-id + access-log middleware | [app/core/middleware.py](app/core/middleware.py) |
| Async SQLAlchemy engine/session | [app/db/session.py](app/db/session.py) |
| Declarative base + column mixins | [app/db/base.py](app/db/base.py) |
| Redis client factory | [app/core/redis.py](app/core/redis.py) |
| Liveness + readiness endpoints | [app/api/health.py](app/api/health.py) |
| App factory + lifespan | [app/main.py](app/main.py) |
| Async Alembic environment | [alembic/env.py](alembic/env.py) |
| Postgres (pgvector) + Redis + app | [docker-compose.yml](docker-compose.yml) |
| Async test setup | [tests/conftest.py](tests/conftest.py) |

The layered structure (`api/` → `services/` → `repositories/`) is created now
even though `services/` and `repositories/` are empty. Deciding the boundaries
before there is code to move is much cheaper than refactoring later.

### Key decisions and why

**1. Config is a typed `Settings` object, never `os.environ`.**
Every knob is declared once with a type and a default. A malformed
`DB_POOL_SIZE=abc` fails at boot with a readable pydantic error instead of
surfacing as a `TypeError` under load. `get_settings()` is `lru_cache`d so the
`.env` file is parsed once per process and the same instance is injectable as a
FastAPI dependency.

**2. Connection details are stored as parts, not as a URL string.**
`database_url` is a `computed_field` assembled from host/port/user/password/db.
This lets docker-compose override *only* `POSTGRES_HOST` (the one thing that
differs between "app in a container" and "app on my laptop") instead of
duplicating an entire URL in two places where they can drift apart.

**3. `postgresql+asyncpg://`, and the pool lives at module scope.**
The driver suffix is the difference between a genuinely async service and one
that blocks the event loop on every query while looking async. The engine — and
therefore the connection pool — is created once per process; building an engine
per request would open a new pool each time and exhaust Postgres' 100-connection
default almost immediately.

**4. `expire_on_commit=False` on the session factory.**
By default SQLAlchemy expires ORM objects after `commit()`, so touching any
attribute triggers a lazy re-`SELECT`. Under asyncio that lazy load raises
`MissingGreenlet`, because there is no way to await inside `__getattr__`. Turning
expiry off lets a service commit and still return the object.

**5. The session dependency owns rollback; services own commit.**
`get_db_session` rolls back on an exception but never commits. Only a service
knows whether two writes belong to the same unit of work — if the dependency
auto-committed, that decision would be taken away from it.

**6. Raw ASGI middleware instead of `BaseHTTPMiddleware`.**
This is the decision most specific to this project. `BaseHTTPMiddleware` runs
the endpoint in a separate anyio task and pipes the response through a memory
object stream. That is invisible for JSON responses but gets in the way of
long-lived streaming responses — which is exactly what Phase 2's SSE `/v1/chat`
endpoint is. Raw ASGI middleware only wraps `send`, so chunks pass straight
through. Choosing this now avoids ripping out working middleware in Phase 2.

**7. One log pipeline for the whole process.**
`structlog` handles our own calls; `ProcessorFormatter` with a
`foreign_pre_chain` pulls uvicorn/SQLAlchemy/Alembic's stdlib records into the
*same* renderer, so there is exactly one log format on stdout. The request id is
propagated with `contextvars`, not function arguments — each request is its own
task, so a `bind_contextvars` at the top of the middleware attaches the id to
every line logged anywhere in that request. JSON in deployed environments (for a
log aggregator), coloured console output locally.

**8. Liveness and readiness are separate endpoints.**
`/health` touches nothing external and answers "is this process alive?".
`/health/ready` checks Postgres and Redis and answers "should I get traffic?".
Conflating them is a genuine outage pattern: if a database check fails liveness,
a 30-second Postgres blip makes the orchestrator kill every *healthy* app
replica, converting a recoverable dependency problem into a restart storm.
Readiness instead removes the pod from the load balancer and lets it rejoin on
its own. The readiness handler also runs both checks even when the first fails,
so the response says *which* dependencies are down.

**9. Migrations run as a deploy step, not in the app's lifespan.**
Running `alembic upgrade head` on startup means N replicas racing to migrate the
same database. In compose it is a separate command before uvicorn; in a real
deployment it would be an init-container or a job.

**10. The pgvector extension is enabled by a migration, not an init SQL script.**
A `docker-entrypoint-initdb.d` script only runs on a fresh volume and does not
exist at all on managed Postgres. A migration is version-controlled and applies
the same way everywhere.

**11. Constraint naming convention set on `MetaData` from day one.**
Without it, Postgres auto-names constraints and Alembic generates migrations
that can't be reversed (it doesn't know the name to drop). It costs one dict now
and is painful to retrofit onto a live database.

**12. `DateTime(timezone=True)` on the timestamp mixin.**
A bare `Mapped[datetime]` maps to `TIMESTAMP WITHOUT TIME ZONE`, which discards
the offset. Phase 6 computes latency and p95 from these columns; a container
running in a different TZ would silently corrupt them.

**13. Multi-stage Dockerfile, non-root runtime user.**
`asyncpg` has C extensions, so a compiler is needed to install it — but only at
build time. The runtime stage copies the finished virtualenv and ships without
`build-essential`. The app runs as uid 1000 rather than root.

**14. `depends_on: condition: service_healthy` with `pg_isready`.**
Postgres binds its port several seconds before it will accept queries, so a
plain TCP check lets the app start too early and crash on first connect.
`pg_isready` tests the thing we actually care about.

### Trade-offs I accepted

- **UUID primary keys** are 16 bytes and hurt B-tree insert locality versus
  `bigserial`. Accepted because ids are handed to API consumers, and sequential
  integers leak how many documents exist and are trivially enumerable.
- **`pool_pre_ping=True`** costs a round trip on every checkout. Accepted
  because it converts "server closed the connection" 500s into a transparent
  reconnect.
- **Phase 1 tests use fakes, not real containers.** They verify wiring, and run
  in 0.1s with nothing installed. Real-container integration tests come in
  Phase 7; both tiers are worth having.

### Interview questions I should be able to answer

1. **Why did you write your own ASGI middleware instead of subclassing
   `BaseHTTPMiddleware`, and what would break if you hadn't?**
   (Expected: `BaseHTTPMiddleware` runs the endpoint in a separate task and
   buffers the response through a memory stream, which interferes with the SSE
   streaming endpoint; raw ASGI middleware just wraps `send`. Bonus: explain
   that middleware is added outermost-last and why the request-id middleware
   needs to be outermost.)

2. **What's the difference between a liveness and a readiness probe, and what
   concretely goes wrong if your liveness probe checks the database?**
   (Expected: liveness → restart the container; readiness → remove from the load
   balancer. A DB check in liveness turns a transient Postgres outage into every
   replica being killed and restarted simultaneously, which usually makes the
   outage worse. Should also be able to say why `SELECT 1` beats a TCP check.)

3. **You set `expire_on_commit=False`. What is the default behaviour, and what
   specific error does it cause in an async app?**
   (Expected: SQLAlchemy expires attributes after commit so the next attribute
   access emits a lazy `SELECT`; in asyncio there's no way to await inside
   attribute access, so it raises `MissingGreenlet`. Bonus: explain why
   SQLAlchemy needs `greenlet` for async at all — the ORM internals are
   synchronous and greenlet bridges them to the async driver.)

4. *(stretch)* **Why is `get_settings()` cached with `lru_cache`, and how would
   you override it in a test?**
   (Expected: parse `.env` once per process, stable object identity for DI;
   override via `app.dependency_overrides[get_settings]` or
   `get_settings.cache_clear()`.)

---

## Phase 2 — Provider abstraction + SSE streaming

### What I built

| Piece | Location |
|---|---|
| Interface + normalized event types | [app/llm/base.py](app/llm/base.py) |
| Provider-neutral error hierarchy | [app/llm/errors.py](app/llm/errors.py) |
| Ollama (default, local, free) | [app/llm/ollama.py](app/llm/ollama.py) |
| OpenAI | [app/llm/openai_provider.py](app/llm/openai_provider.py) |
| Anthropic | [app/llm/anthropic_provider.py](app/llm/anthropic_provider.py) |
| Lazy provider registry | [app/llm/registry.py](app/llm/registry.py) |
| Orchestration | [app/services/chat_service.py](app/services/chat_service.py) |
| SSE endpoint | [app/api/v1/routes/chat.py](app/api/v1/routes/chat.py) |
| Wire schemas | [app/schemas/chat.py](app/schemas/chat.py) |

### The core problem this phase solves

The three providers stream in genuinely different shapes:

| | Transport | Text arrives as | Token counts |
|---|---|---|---|
| Ollama | NDJSON | `message.content` per line | final line: `prompt_eval_count` / `eval_count` |
| OpenAI | SSE | `choices[0].delta.content` | **absent by default** — must opt in |
| Anthropic | SSE (typed events) | `content_block_delta` | split across `message_start` + `message_delta` |

Every provider collapses that into the same two-event stream: zero or more
`TextDelta`, then exactly one `StreamDone` carrying usage. That invariant —
*exactly one, always last* — is what lets Phase 4's cache and Phase 6's metrics
hook in at one place regardless of who served the request.

### Key decisions and why

**1. Official SDKs for OpenAI and Anthropic; raw httpx for Ollama.**
The SDKs handle retries with backoff, auth, and streaming-event typing — code I
would otherwise write and maintain badly. Ollama has no official Python SDK and
its API is plain NDJSON, so httpx is the right tool there. The trade-off I
accepted: two more dependencies, and each SDK has its own release cadence to
track. The alternative (httpx for all three, one HTTP client, one timeout
policy) is defensible and is roughly what LiteLLM does — I judged
robustness-for-free worth more than uniformity here.

**2. Only `stream_chat` is abstract; `complete()` is derived from it.**
A new provider implements one method and gets both APIs. More importantly,
streaming and non-streaming can't drift apart, because there is only one code
path — the batch version is literally the stream accumulated.

**3. A tagged union (`TextDelta | StreamDone`), not one event class with
optional fields.** After `isinstance(ev, StreamDone)` the type checker knows
`usage` exists. With a single class carrying `Optional[Usage]`, every access
would need a None check that can never actually be None.

**4. `stream_chat` is a plain `def` returning an `AsyncIterator`, not an
`async def`.** Calling an async generator function executes *none* of its body,
so an unreachable provider wouldn't raise until the first `__anext__()`. Making
the signature a plain `def` keeps that visible, and the route depends on it (see
decision 8).

**5. Errors are translated into a provider-neutral hierarchy at the boundary.**
The router never imports `openai` or `anthropic` to catch anything. Adding a
fourth provider must not require editing the web layer's except-clauses.
`ProviderAuthError` maps to **502, not 401** — deliberately. A bad API key means
*docXpo* is misconfigured; returning 401 would tell our caller to re-authenticate,
which fixes nothing.

**6. Providers are built lazily, not at startup.** Constructing the OpenAI
provider raises if `OPENAI_API_KEY` is missing, so eager construction would make
the app refuse to boot unless you held keys for all three. Lazily, Ollama-only
development works with an empty `.env`. The registry uses double-checked locking
so two concurrent first-requests can't build two connection pools.

**7. `stream_options={"include_usage": True}` on OpenAI is mandatory, not
optional.** Without it a streaming response carries no token counts at all —
`usage` is null on every chunk. Phase 6 computes cost from those numbers, so
omitting it would silently produce a metrics endpoint that reports $0.00
forever. It costs one extra final chunk with an empty `choices` list, which is
why the parse loop tolerates that shape.

**8. The stream is "primed" before the response is returned.**
Once a `StreamingResponse` begins, the `200 OK` and headers are on the wire and
cannot be recalled. So the route pulls the *first* event before constructing the
response: a dead Ollama or a bad key becomes a real `503`/`502`, not
`200 OK` followed by an error event that no HTTP client treats as a failure.
Failures *after* the first token still have to be reported in-band as an SSE
`error` event — that asymmetry is inherent to streaming, not a shortcut.

**9. SSE `data:` is JSON-encoded, always.** SSE is newline-delimited; a token
containing a literal newline written raw would terminate the frame early and
desynchronise every subsequent event. `json.dumps` escapes it to `\n`. There is
a test that exists solely to pin this down.

**10. `X-Accel-Buffering: no`.** Without it nginx buffers the response until its
buffer fills, and the "streaming" endpoint delivers everything in one burst at
the end. This is the single most common reason SSE works locally and fails in
production.

**11. `split_system()` lives in the abstraction, not in the caller.** Ollama and
OpenAI take the system prompt as a message role; Anthropic rejects that role and
requires a top-level `system` parameter. Multiple system messages are joined
rather than dropped, so nothing the caller sent is silently discarded.

**12. SSE over WebSockets.** Traffic is one-directional. SSE is plain HTTP — it
traverses proxies with no upgrade handshake and browsers reconnect on their own.
A WebSocket would buy bidirectionality we don't need.

### Trade-offs I accepted

- **Anthropic's `max_tokens` caps thinking *plus* visible text** on current
  models, and thinking is on by default on Opus 5. A value tuned for the answer
  alone can truncate the response. The default (1024) is deliberately modest;
  raise it for that provider.
- **Latency is measured to the last token**, not the first. Time-to-first-token
  is the better UX metric; Phase 6 will record both.
- **`TextDelta` is a fragment, not a token.** Providers chunk differently, so
  the `chunks` count in logs is not a token count — token counts come from
  `StreamDone.usage`.

### Interview questions I should be able to answer

1. **Your `/v1/chat` endpoint returns `200` and then discovers the provider is
   down. How do you report that, and why can't you just return a 503?**
   (Expected: headers and status are already flushed once streaming starts, so
   the only channel left is an in-band SSE `error` event. Then the follow-up:
   how do you *avoid* that for connection-level failures — pull the first event
   before returning the `StreamingResponse`, so those become real status codes.)

2. **Why is the SSE `data:` field JSON-encoded, and what breaks if it isn't?**
   (Expected: SSE frames are newline-delimited; a raw token containing `\n`
   splits one frame into two and corrupts every event after it. Bonus:
   `X-Accel-Buffering: no` and why streaming silently breaks behind nginx.)

3. **Three providers, three different streaming formats. What exactly does your
   interface normalize, and what would adding a fourth provider require?**
   (Expected: name the concrete differences — NDJSON vs SSE, where usage lives,
   system-prompt-as-role vs top-level param, OpenAI needing `include_usage` to
   report tokens at all. Adding one = implement `stream_chat` + error
   translation, register it; nothing in the router or service changes.)

4. *(stretch)* **Why is `complete()` concrete instead of abstract, and why are
   providers built lazily rather than at startup?**
   (Expected: one code path so batch/stream can't diverge; lazy construction so
   a missing `OPENAI_API_KEY` doesn't stop the app booting for a user of another
   provider — plus double-checked locking so concurrent first-requests build one
   pool. See the amendment below: the *default* provider is now checked eagerly.)

### Amendment — Anthropic as the real default, no fake fallback

Made after Phase 2, at my request: run against a real Anthropic key rather than
a local model.

**What changed**

- `LLM_PROVIDER` → **`DEFAULT_PROVIDER`**, defaulting to `anthropic`.
- `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-6`.
- Deleted `tools/fake_ollama.py`, the mock NDJSON server used to demo streaming
  without a model. Nothing in the app can now serve synthetic tokens.
- Added `LLMProvider.check_credentials()` and eager validation of the *default*
  provider during startup.

**Why the default provider is now validated eagerly, reversing decision 6.**
Phase 2 built every provider lazily so a missing key surfaced on first use. That
is the wrong trade once there is no fallback: a deployment whose LLM is
unreachable is broken, and it should not accept traffic while pretending
otherwise. Now the *default* provider is constructed and verified at startup —
if that fails, `lifespan` raises, uvicorn exits non-zero, and an orchestrator
halts the rollout. Non-default providers stay lazy, so an Anthropic deployment
still doesn't need an OpenAI key to boot.

**Why `models.retrieve()` and not a one-token generation.**
The check is `GET /v1/models/{id}` — free, non-generative, and it validates two
things at once: that the key is accepted, and that the configured model ID
actually exists for this account. A typo like `claude-sonnet-4.6` fails at boot
rather than on a user's first request. A test generation would bill tokens on
every deploy and every container restart.

**The trade-off this costs.** Startup now depends on Anthropic being reachable,
so a network blip blocks a deploy. `VALIDATE_PROVIDER_ON_STARTUP=false` skips
the *network call* for offline work — but it deliberately does **not** allow
booting with no key at all, which would reintroduce the silent-failure mode this
change exists to remove. There's a test pinning that distinction.

**What I did *not* remove.** The Ollama and OpenAI providers, and the
`FakeProvider` in `tests/`. "Instead of Ollama" is satisfied by making Anthropic
the default; deleting the other providers would gut the multi-provider
abstraction that is the point of Phase 2. The test double is a test double —
removing it would mean the SSE plumbing could only be tested by spending money
on live API calls. Say the word if you want the Ollama/OpenAI code deleted
outright.

**Interview question this adds:** *Your app calls a paid API. Why validate
credentials at startup instead of on first use, and what does that cost you?*
(Expected: fail-fast so a bad rollout halts instead of failing per-request;
cost is a hard startup dependency on an external service, mitigated by an
opt-out flag that skips the call but not the key requirement. Bonus: why a free
metadata endpoint beats a test generation.)

### Amendment 2 — Gemini added, and made the default

Driven by cost: the Anthropic account ran out of credits, and Gemini's free tier
is the most generous of the hosted providers.

**What changed**

- New [app/llm/gemini_provider.py](app/llm/gemini_provider.py) using the
  official `google-genai` SDK, model `gemini-flash-latest`.
- `DEFAULT_PROVIDER` now defaults to `gemini`. Anthropic, OpenAI, and Ollama are
  untouched and reachable by env var or per-request override.

**Gemini was the real test of the abstraction — it disagrees with the others in
four separate ways**, and none of them leaked upward:

| | Gemini | Everyone else |
|---|---|---|
| Assistant role | `"model"` | `"assistant"` |
| System prompt | `config.system_instruction` | message role (Ollama/OpenAI) or top-level `system` (Anthropic) |
| Output cap | `max_output_tokens`, nested in a config object | `max_tokens` |
| Usage | **cumulative on every chunk** | final line only (Ollama) / one opt-in final chunk (OpenAI) / split across two events (Anthropic) |

The cumulative-usage one is the dangerous one: summing across chunks instead of
overwriting would have reported ~3× the real token count and silently inflated
every cost number in Phase 6. There's a dedicated test pinning it.

Also worth noting: `generate_content_stream` is a *coroutine returning* an async
iterator, so it needs `await` before `async for` — unlike the Anthropic and
OpenAI stream helpers. That asymmetry is exactly the kind of thing the interface
exists to absorb.

**Bug found while verifying against the live API.** Google returns **400
INVALID_ARGUMENT** for an invalid API key, not 401. My first error mapping sent
that to `ProviderBadRequest` → HTTP 400, which would have blamed the *caller*
for what is actually our own misconfiguration. Now detected by message and
mapped to `ProviderAuthError` → 502, consistent with the other providers. This
only surfaced because I tested with a real invalid key rather than a mock.

**A limitation I did not fix.** `check_credentials()` uses a free metadata
endpoint, so it verifies the key is *accepted* but not that quota remains. A
zero-credit Anthropic key and a zero-quota Gemini key both still boot cleanly and
fail on the first real request — which happened with *both* providers in
practice. Closing that gap requires a real 1-token generation at startup, which
costs money on every restart and every `--reload`. Deliberately left open.

**Model choice: `gemini-flash-latest`, not `gemini-2.0-flash`.** The requested
model turned out to be unusable on a newly-created API key: `gemini-2.0-flash`
and `gemini-2.0-flash-lite` return `429` with `limit: 0` (no free-tier quota
granted at all, as distinct from quota exhausted), and `gemini-2.5-flash` returns
`404 "no longer available to new users"`. Of the candidates tried, only the
rolling `gemini-flash-latest` alias generated successfully — it resolved to
`gemini-3.6-flash` at time of writing. The trade-off of an alias is that the
underlying model can change under you, which matters for reproducibility; pin a
concrete ID once you know which one your account can actually serve.

**Interview question this adds:** *You support four LLM providers behind one
interface. Give a concrete example of something one provider does differently,
and explain where in your code that difference is absorbed.*
(Expected: name a real one — Gemini's `"model"` role, or cumulative usage per
chunk — and point at the provider class as the single place it's handled, with
the normalized `StreamEvent` contract as what everything above sees. Strong
answer also mentions that the error hierarchy is normalized too, so the router
never imports a vendor SDK.)

---

## Phase 3 — RAG

### What I built

| Piece | Location |
|---|---|
| Embedding interface + Gemini/OpenAI impls | [app/llm/embeddings.py](app/llm/embeddings.py) |
| Recursive chunker | [app/services/chunking.py](app/services/chunking.py) |
| Document / Chunk tables + HNSW index | [app/db/models/document.py](app/db/models/document.py) |
| Vector search | [app/repositories/document_repository.py](app/repositories/document_repository.py) |
| Ingest pipeline | [app/services/document_service.py](app/services/document_service.py) |
| Retrieve + grounded generation | [app/services/rag_service.py](app/services/rag_service.py) |
| Endpoints | [documents.py](app/api/v1/routes/documents.py), [rag.py](app/api/v1/routes/rag.py) |

`repositories/` finally has a reason to exist: it is the only layer that knows SQL.

### Chunking: ~1200 chars, 200 overlap, recursive

**Why chunk at all** — two distinct reasons. Embedding models have an input
limit, yes; but the bigger one is that *one vector per document is a bad
retrieval unit*. An embedding averages everything in its input, so a 40-page
manual becomes a vector meaning "this is a manual" that matches nothing
specifically.

**Recursive splitting** tries separators in descending semantic strength —
paragraph, line, sentence, space, hard cut. Anything still oversized is
re-split by the next-weaker separator. Chunks therefore break at paragraph
boundaries when possible, sentence boundaries when necessary, and mid-word
essentially never. Naive fixed-width slicing cuts mid-sentence and embeds badly.

**~1200 chars (~300 tokens)** sits near one or two well-formed paragraphs — the
unit at which a document usually makes a single point. Too small and a chunk
loses the context that makes it meaningful ("It supports 512 connections" is
useless without knowing what "it" is). Too large and the embedding averages over
too many ideas, stops being specific, and wastes the generator's context window.

**200 char overlap (~17%)** exists for one failure mode: a fact that straddles a
boundary is otherwise destroyed. Split between "The retry limit is" and "set to
5" and *neither* chunk can answer the question. The cost is real — ~17% more
rows, embedding calls and storage, plus near-duplicate chunks that can occupy
two top-k slots. 10-20% is the usual sweet spot.

### Key decisions

**1. Embeddings are a separate interface from `LLMProvider`.** Anthropic has no
embedding model at all, so folding `embed()` into `LLMProvider` would force
`AnthropicProvider` to implement a method it can never satisfy. Separate
interfaces mean you can legitimately chat on Anthropic and embed on Gemini.

**2. 768 dimensions, not the native 3072.** pgvector's HNSW and IVFFlat indexes
both **refuse columns wider than 2000 dimensions** — a 3072-dim column could only
ever be sequentially scanned. 768 also cuts storage 4x (3KB vs 12KB per chunk).

**3. Embeddings are re-normalized after dimension reduction.** `gemini-embedding-001`
is a Matryoshka model: its native output is unit-normalized, but truncating to
768 destroys that — **measured L2 norm 0.587, not 1.0**. Cosine distance would
still rank correctly, but absolute similarity scores stop being comparable, which
makes any fixed threshold meaningless. Phase 4's semantic cache is built entirely
on such a threshold, so this had to be right now.

**4. Asymmetric embeddings.** Gemini encodes passages (`RETRIEVAL_DOCUMENT`) and
questions (`RETRIEVAL_QUERY`) differently on purpose. Using one encoding for both
silently costs recall. OpenAI is symmetric, so its embedder accepts and ignores
the `task` argument — which is what keeps callers provider-agnostic.

**5. HNSW, not IVFFlat.** IVFFlat clusters *existing* rows to build its lists, so
creating it in a migration against an empty table gives poor recall until
rebuilt — a footgun when documents arrive after deploy. HNSW builds incrementally
and is correct from row one. Costs a slower build and more memory; irrelevant at
this scale.

**6. The index operator class must match the query operator.** The index is
`vector_cosine_ops` and the query uses `<=>`. Using `<->` (L2) against a cosine
index makes Postgres *silently* ignore the index and sequentially scan. Related:
the similarity floor is applied **after** the ordered fetch, because a `WHERE`
on the distance expression would also defeat the index.

**7. Context goes in a system message, not the user turn.** It keeps the user's
question as the last thing the model reads, and the Phase 2 abstraction already
routes system messages correctly per vendor (Anthropic top-level `system`,
Gemini `system_instruction`, a role for the rest).

**8. `/rag/retrieve` exists separately from `/rag/query`.** The two halves of RAG
fail differently — a bad answer is either bad retrieval or bad generation.
Inspecting retrieval with no tokens spent is what makes chunk size and `top_k`
tunable rather than guesswork.

### The measurement that changed a default

I set `rag_min_similarity` to 0.35 by intuition. Measured against the real corpus:

| Query | Top similarity |
|---|---|
| On-topic, answerable | 0.672 |
| On-topic, **not** in the corpus | 0.626 |
| Complete nonsense | **0.542** |

A 0.35 floor can never fire. But the range is compressed enough that no
threshold cleanly separates "unrelated" (0.54) from "relevant" (0.63-0.67)
either. Raised to **0.50**, which usefully drops off-topic *documents* from
on-topic queries — an unrelated gardening doc scoring 0.481 no longer occupies a
top-k slot — while accepting that pure nonsense still gets through.

The real defence against answering from noise is the system prompt, and it
works: asked something on-topic but absent from the corpus, the model replied
*"the provided text does not contain information about what port the gateway
listens on."* The floor is a token-saving safety net, not a precision filter.

### Verified end-to-end

Ingest, chunk, embed, store in pgvector, retrieve, grounded answer — against a
real Gemini key and real Postgres. Ranking correct (0.672 relevant vs 0.481
irrelevant); citation `[1]` resolved to the right source; refusal behaviour
correct. 67 tests pass.

### Interview questions I should be able to answer

1. **Why overlap chunks at all, and how did you pick 200 characters?**
   (Expected: a boundary-spanning fact is destroyed without it — neither chunk
   answers the question. Cost is ~17% more rows/embeddings/storage plus
   near-duplicates competing for top-k slots. Bonus: why recursive separator
   splitting beats fixed-width slicing.)

2. **Your embeddings are 3072-dim natively but you store 768. Why, and what
   had to change as a result?**
   (Expected: pgvector's index ceiling is 2000 dims, so 3072 is unindexable.
   The consequence: truncating a Matryoshka embedding breaks unit
   normalization — measured 0.587 — so vectors must be re-normalized or
   absolute similarity scores become incomparable and thresholds meaningless.)

3. **You retrieve top-k by cosine similarity. What stops the model answering
   from its own knowledge when retrieval returns nothing useful?**
   (Expected: a similarity floor *and* a system prompt that makes "not in the
   documents" an allowed answer. Strong answer notes the floor is weak because
   the score range is compressed — cite the 0.54 vs 0.63 measurement — and that
   the prompt does the real work. Bonus: why the floor is applied after the
   ORDER BY rather than as a WHERE.)

4. *(stretch)* **Why is embedding a separate interface from your LLM provider?**
   (Expected: Anthropic has no embedding model; one combined interface forces an
   unimplementable method. Also enables chat-on-one-vendor, embed-on-another.)

---

## Phase 4 — Semantic cache

### What I built

| Piece | Location |
|---|---|
| Cache (lookup, store, invalidation) | [app/services/semantic_cache.py](app/services/semantic_cache.py) |
| Wired into the RAG query path | [app/api/v1/routes/rag.py](app/api/v1/routes/rag.py) |
| Corpus-version invalidation | [app/services/document_service.py](app/services/document_service.py) |
| `GET`/`DELETE /v1/rag/cache` | [app/api/v1/routes/rag.py](app/api/v1/routes/rag.py) |

An exact-match cache is nearly useless for natural language — "what is the retry
limit?" and "what's the retry limit" are different strings. A semantic cache
compares *meaning* by cosine similarity and serves the stored answer when the
incoming question is close enough to one already answered.

### The threshold trade-off — measured, not guessed

This is the one number that defines the feature. Measured query-to-query cosine
on real embeddings:

| Case | Cosine |
|---|---|
| identical | 1.0000 |
| typo | 0.9933 |
| paraphrase | 0.9853 |
| **negated — "does NOT use"** | **0.9752** |
| looser paraphrase | 0.9169 |
| same topic, different ask | 0.8666 |
| same nouns, opposite intent | 0.8008 |
| different topic | 0.5385 |
| unrelated | 0.4200 |

The trade-off runs both ways, and the two failures cost very different amounts:

* **Threshold too low → false hits.** The cache confidently answers a
  *different* question. This is the expensive failure: silently wrong output,
  no error, no way for the caller to tell.
* **Threshold too high → few hits.** The cache costs a little and saves
  nothing. Cheap, boring, safe.

Because the downside is asymmetric, the default is conservative: **0.98**.

**The finding that set it: a negated question scored 0.9752 — higher than a
legitimate looser paraphrase at 0.9169.** "What threshold does the cache use?"
and "What threshold does the cache NOT use?" are semantic opposites that embed
almost identically, because embeddings encode *topic* far more strongly than
*polarity*. Any threshold loose enough to catch the 0.917 paraphrase would also
serve the inverse answer to the negated question.

No threshold fixes that — it only trades hit rate against the probability of
being wrong. 0.98 clears the measured negation by 0.005, which is a thin margin
and honestly stated as such. A test pins the default above 0.9752 so nobody
lowers it casually.

Also worth noting: query-to-query similarity has a far wider usable range
(0.42–1.00) than the query-to-document similarity from Phase 3 (0.54–0.67).
That is the difference between comparing two *symmetric* query encodings and
comparing a query encoding against a document encoding.

### Key decisions

**1. Brute-force scan, not RediSearch.** Plain `redis:7-alpine` has no vector
index; getting one means the much larger `redis-stack` image. Instead the
namespace's vectors are fetched and scored with a single numpy matmul. That is
O(n), which is why the cache is **bounded** — 500 entries × 768 dims is one
(500, 768) @ (768,) product, roughly 0.3ms, against an LLM call costing seconds.
Past ~10k entries the right move is RediSearch's HNSW, or the pgvector index we
already have.

**2. The dot product *is* the cosine similarity — because of Phase 3.** Every
embedding is unit-normalized by the embedding provider, and for unit vectors
cos(a,b) = a·b. No division by norms anywhere. This is the payoff for fixing
normalization in Phase 3: if that invariant broke, every score here would be
silently wrong.

**3. The cache costs zero extra embedding calls.** RAG already embeds the
question for vector search, so `embed_query()` was extracted and the same vector
feeds both the cache lookup and the retrieval. A cache that embedded on its own
would spend an API round trip just to discover it had missed.

**4. Namespaced by provider + model + corpus version.** A Gemini answer must
never be served to an Anthropic request. And since a RAG answer is derived from
the document set, adding or deleting a document bumps a counter that changes the
namespace — orphaning every stale answer in **O(1)**, with no scan-and-delete.
The orphans are not swept; they simply expire.

**5. Namespace on the requested model, display the resolved one.** A request for
`gemini-flash-latest` resolves to `gemini-3.6-flash`. The namespace keys on what
the caller asked for (stable), but the stored entry records what actually
generated the answer, so a hit reports the same model a fresh generation would.
I noticed this only because the first live test showed two different model names
for the same question.

**6. Only complete answers are cached.** Storing on the success path only —
a partial answer from a client disconnect or mid-stream error would otherwise be
served, truncated, to every future match.

**7. A hit streams in the same SSE shape as a generation.** The client receives
`sources`, `token`, `done` exactly as usual, with `cached: true` on the done
event. The answer arrives as one token frame because it is already complete —
re-chunking it to fake incremental typing would add latency for cosmetics.
`sources` is empty on a hit: nothing was retrieved, and inventing citations we
did not look up would be dishonest.

**8. Chat is deliberately not cached.** `/v1/chat` is multi-turn, so the right
answer depends on conversation history, not just the last message. Two identical
questions in different conversations should get different answers.

### Verified against live Redis and Gemini

| Query | Result |
|---|---|
| cold (seed) | generated, 2470ms, 649 tokens |
| identical | **HIT** sim 1.0000, 3.41ms, 649 tokens saved |
| paraphrase | **HIT** sim 0.9853, 0.94ms, 649 tokens saved |
| "How do I *disable* the cache?" | miss → generated |
| different topic | miss → generated |

~1000x faster on a hit, and the same-nouns-different-intent question correctly
missed. 82 tests pass.

### Interview questions I should be able to answer

1. **How did you choose the similarity threshold, and what does getting it
   wrong cost in each direction?**
   (Expected: too low serves a confidently wrong answer to a different question
   — silent, unloggable; too high just wastes a little effort. Asymmetric, so
   default conservative. The strong answer cites the measurement: a *negated*
   question scored 0.975, above a real paraphrase at 0.917, because embeddings
   capture topic much better than polarity — so no threshold is actually safe,
   it only trades hit rate for error probability.)

2. **Your cache does a linear scan. Why is that acceptable, and when does it
   stop being acceptable?**
   (Expected: bounded entry count + a vectorised matmul makes it ~0.3ms against
   a multi-second LLM call, and plain Redis has no vector index. It stops
   scaling around 10k entries, at which point RediSearch HNSW or pgvector.
   Bonus: why the dot product suffices — unit-normalized vectors from Phase 3.)

3. **A user uploads a new document. What happens to previously cached answers,
   and why did you do it that way?**
   (Expected: a corpus-version counter is part of the cache namespace, so a
   bump orphans every stale answer in O(1). The alternative — scanning and
   deleting matching keys — makes ingestion O(cache size). Orphans expire via
   TTL rather than being swept.)

4. *(stretch)* **Why does the cache add no embedding cost, and why is chat not
   cached?**
   (Expected: RAG already embeds the query for retrieval, so the vector is
   shared. Chat is multi-turn — the correct answer depends on history, so
   matching on the last message alone would be wrong.)

---

## Phase 5 — Rate limiting

### What I built

| Piece | Location |
|---|---|
| Token bucket + Lua script | [app/core/rate_limit.py](app/core/rate_limit.py) |
| API-key identity | [app/core/auth.py](app/core/auth.py) |
| ASGI middleware | [app/core/rate_limit_middleware.py](app/core/rate_limit_middleware.py) |
| API-key field in the console | [app/web/index.html](app/web/index.html) |

### Why a token bucket

| Algorithm | Memory | Burst behaviour |
|---|---|---|
| Fixed window | 1 counter | Allows **2x** at a boundary: 60 requests at 11:59:59 and 60 more at 12:00:00 both pass |
| Sliding window log | O(n) timestamps | Exact, but stores every request |
| **Token bucket** | **2 numbers** | Smooth refill, bounded burst, O(1) |

A bucket holds `capacity` tokens and refills continuously at `rate` per second.
An idle client can burst up to capacity, then settles to the sustained rate —
which matches how real clients actually behave (bursty, then quiet) far better
than a hard per-minute cap. Defaults: 60 rpm sustained, 20 burst.

### Key decisions

**1. The arithmetic is a Lua script, and that is the point.** Refill → check →
decrement must be **atomic**. As separate GET/SET calls, two concurrent requests
both read `tokens=1`, both conclude they may proceed, and both write `tokens=0`
— the limit is silently exceeded under exactly the load it exists to control.
Redis runs a Lua script atomically on its single-threaded core, so nothing
interleaves. This is the whole reason the logic lives in Redis and not in Python.

**2. Middleware, not a FastAPI dependency.** Two reasons. A dependency cannot
reliably attach headers to a `StreamingResponse` the route builds and returns
itself — and `X-RateLimit-*` on *successful* streaming responses is most of the
value. And a dependency is opt-in: forgetting to declare it silently leaves a
route unmetered. An explicit exempt list fails safe in the other direction.

**3. Raw ASGI middleware**, for the same reason as Phase 1's
`RequestContextMiddleware`: `BaseHTTPMiddleware` buffers the body through a
memory stream, which is exactly wrong for the SSE endpoints this most needs to
protect.

**4. The limiter sits *inside* the request-context middleware.** Middleware is
added outermost-last, so `RequestContextMiddleware` wraps the limiter. That
means a 429 still gets a request id and still appears in the access log. If the
limiter were outermost, rejected traffic would be invisible — precisely the
traffic you most want to see.

**5. `/health` and `/` are exempt.** A probe rejected because a noisy client
drained a shared bucket would fail the orchestrator's health check and trigger a
rolling restart — converting a rate-limit event into an outage.

**6. Keys are fingerprinted, never stored raw.** The Redis bucket key is
`rl:key:f0151b84a66c` (SHA-256, first 12 hex chars). A leaked log line or a
`redis-cli --scan` exposes no credential. Comparison uses `hmac.compare_digest`
rather than set membership, because `in` short-circuits on the first differing
byte and leaks key material through timing.

**7. Rate-limit headers on every response, not just 429s.** A client should be
able to see `X-RateLimit-Remaining` falling and slow down *before* being
rejected, rather than discovering the limit by hitting it. `Retry-After` is
added only on rejection — that is what the HTTP spec defines for 429 and what
proxies honour — and is rounded **up**, because a `Retry-After: 0` invites an
immediate retry guaranteed to fail again.

**8. Empty `API_KEYS` means open mode, metered by IP.** This keeps local dev and
the browser console working with an empty `.env`; setting any key switches on
enforcement. The limitation is stated honestly in the code: behind a proxy every
request carries the proxy's IP unless `X-Forwarded-For` is handled, and that
header is client-controlled and trivially spoofed. IP metering is a development
convenience, not a security control.

**9. Fail-open by default, and deliberately broad.** The `except` catches
`Exception`, not just `RedisError` — fail-open only means anything if it covers
*every* way this can break, including a Lua bug or a misbehaving client. A rate
limiter that can crash the request path is worse than no rate limiter. The
trade-off is explicit: during a Redis outage the service is unprotected.
`RATE_LIMIT_FAIL_OPEN=false` inverts it for deployments where an unmetered flood
costs more than an outage.

I found this one *because* a test failed: an `AttributeError` from a fake client
was not a `RedisError`, so fail-open never engaged and the request 500'd. That
is exactly the production failure mode, surfaced by accident.

**10. Buckets get a TTL.** `PEXPIRE` set to the full-refill time. An idle bucket
is indistinguishable from a fresh one, so keeping it wastes memory — without
this, Redis accumulates a key per API key forever.

### Verified against live Redis

| Check | Result |
|---|---|
| Headers on a normal 200 | `limit 60`, `remaining 19`, `reset 1` |
| Burst exhaustion | 21x `200` then `429` (21 because a token refilled mid-loop) |
| 429 payload | `Retry-After: 1`, `X-RateLimit-Remaining: 0`, typed JSON body |
| Exempt while limited | `/v1/*` → 429, but `/health`, `/health/ready`, `/`, `/docs` → 200 |
| Refill | blocked, then all 200 again after 3s |
| No key / wrong key | 401 + `WWW-Authenticate: Bearer realm="docXpo"` |
| Valid key, both header styles | `X-API-Key` and `Authorization: Bearer` → 200 |
| **Per-key isolation** | alpha exhausted → 429; beta unaffected → 200 200 200 |
| Redis contents | `rl:key:f0151b84a66c` — fingerprints only |

104 tests pass.

### Interview questions I should be able to answer

1. **Why is your token-bucket logic a Lua script instead of Python calling
   GET/SET?**
   (Expected: atomicity. Read-modify-write across separate round trips races —
   two concurrent requests both see the last token and both proceed, so the
   limit fails under exactly the concurrency it exists for. Redis executes Lua
   atomically on a single thread. Bonus: why `register_script` is safe across a
   Redis restart — redis-py falls back to EVAL when the SHA is not cached.)

2. **Token bucket vs fixed window vs sliding window — why did you pick this one?**
   (Expected: fixed window allows 2x at the boundary; sliding window log is
   exact but O(n) memory per key; token bucket is O(1) — two numbers — and
   permits a bounded burst, which matches real client behaviour. Bonus: burst
   and sustained rate are separate knobs, and burst == rpm gives a strict cap.)

3. **Middleware or dependency, and what happens to your rate-limit headers on a
   streaming response?**
   (Expected: dependency can't reliably set headers on a `StreamingResponse` the
   route returns itself, and is opt-in so a forgotten route goes unmetered.
   Raw ASGI middleware wraps `send` and injects headers into
   `http.response.start`, which works for streamed bodies. Bonus: why the
   limiter is registered *inside* the logging middleware — so 429s still get
   logged.)

4. *(stretch)* **What happens when Redis goes down, and is that the right
   choice?**
   (Expected: fails open — availability over protection — and the `except` is
   deliberately broad so any limiter bug can't take down the request path.
   Wrong choice when the upstream cost of an unmetered flood exceeds an outage;
   `fail_open=False` inverts it. Bonus: why raw keys are never used as Redis
   keys, and why `compare_digest` over `in`.)

---

## Phase 6 — Observability

### What I built

| Piece | Location |
|---|---|
| Pricebook (cost estimation) | [app/core/pricing.py](app/core/pricing.py) |
| `request_logs` table | [app/db/models/request_log.py](app/db/models/request_log.py) |
| Recorder + aggregation | [app/services/metrics_service.py](app/services/metrics_service.py) |
| `/v1/metrics`, `/v1/metrics/recent` | [app/api/v1/routes/metrics.py](app/api/v1/routes/metrics.py) |

This is where the earlier phases converge onto one row: the **request id** from
Phase 1's middleware, the **provider/model and token counts** from Phase 2, the
**cache verdict** from Phase 4, and the **API-key fingerprint** from Phase 5.
One query can now answer "what did this key cost us this week, and how often did
the cache save a call?".

### Key decisions

**1. Cost is stored, not computed on read.** Prices change. If `/metrics`
multiplied historical token counts by *today's* rate, last month's spend would
silently change whenever a vendor adjusted pricing — and a cost report that
rewrites its own history is worse than none. Cost is calculated once, at record
time, with the rate then in effect.

**2. Unknown models cost zero *and say so*.** A model with no pricebook entry
records `cost_usd = 0` and `priced = false`, and `/metrics` reports
`unpriced_requests`. Guessing a rate produces a number that looks authoritative
and is wrong; reporting zero silently understates spend. The flag makes an
unpriced model show up as a gap rather than as free.

**3. `Numeric`, never `float` — for money and for latency.** `0.1 + 0.2 != 0.3`
in binary floating point, and summing thousands of sub-cent per-request costs is
exactly where that compounds. Costs are stored at 8dp because a single cheap
request can cost a fraction of a cent, and 4dp would floor thousands of them to
zero.

**4. The recorder opens its own DB session.** Recording happens *after* the
response body has finished streaming, by which point FastAPI has torn down the
request's dependencies — including the `AsyncSession` the route was handed.
Reusing it would fail intermittently and confusingly. `record()` takes a
short-lived session from the factory instead.

**5. A failed metrics write is swallowed.** Observability must never break the
thing it observes. If Postgres is down the user's answer has already been
delivered; refusing to return it because a telemetry row could not be filed
would turn a monitoring outage into a service outage. Logged loudly, dropped.

**6. Recording happens in the stream's `finally`, after the last frame.** So it
adds nothing to what the user waits for. It runs on the error path too — a
partial stream is exactly the kind of request you want counted.

**7. TTFT is stored separately from total latency.** For a streaming endpoint,
time-to-first-token is what a user actually feels; total latency is what
capacity planning needs. They are different numbers and collapsing them loses
the one you wanted. TTFT is null on cache hits, because nothing streamed.

**8. A cache hit records what it *saved*, not what it spent.** `cost_usd = 0`
(nothing was spent) and `cost_saved_usd` = what the stored answer originally
cost. This is what turns Phase 4's cache from "feels fast" into a line item.

**9. `percentile_cont` for p95 — exact, and knowingly so.** Postgres sorts the
window and interpolates, which is correct but O(n log n) over the window. Fine
at this scale. At high volume the usual move is a streaming approximation
(t-digest / HdrHistogram) or pre-aggregated buckets, trading ~1% error for O(1)
memory. Worth knowing that is the trade being made rather than discovering it
at scale.

**10. Longest-prefix model matching.** `claude-sonnet-4-6-20251114` inherits the
`claude-sonnet-4-6` rate, so dated snapshots do not each need an entry. Pricing
is also override-able from config (`PRICING_OVERRIDES`), because correcting a
stale price should not require a code edit and redeploy.

**11. JSON, not Prometheus exposition format.** This is meant to be read by a
human debugging cost and latency, and the numbers that matter (USD, hit rate)
are aggregates over a window rather than the monotonic counters a scraper wants.
A `/metrics/prometheus` alongside it would be straightforward if one were needed.

### Verified end-to-end

Three real requests (cold generate → cache hit → chat), then `/v1/metrics`:

```
requests 3 | cache_hits 1 | cache_hit_rate 0.3333 | errors 0
tokens     in 607, out 289, total 896
cost_usd   spent $0.000905, saved_by_cache $0.000695, unpriced 0
latency_ms p50 1733.8, p95 2788.2, p99 2881.9
```

Row level confirmed in Postgres: request ids correlate with the access log,
`principal` recorded (`ip:172.19.0.1` — open mode), `ttft_ms` populated for
generated requests and **null for the cache hit**, `priced = true` throughout.

118 tests pass.

### Interview questions I should be able to answer

1. **Why is cost stored on the row instead of computed when `/metrics` is
   queried?**
   (Expected: prices change, so computing on read silently rewrites historical
   spend. Store the figure calculated with the rate in effect at the time.
   Bonus: why unknown models are flagged `priced = false` rather than recorded
   as $0 — a silent zero understates spend and looks the same as a genuinely
   free local model.)

2. **Where do you write the metrics row, and what happens if that write fails?**
   (Expected: in the stream's `finally`, after the last frame is delivered, so
   it costs the user nothing; using a *fresh* session because the request's
   dependencies are already torn down by then. A failure is logged and
   swallowed — telemetry must not break the request it is measuring.)

3. **How do you compute p95, and when does that approach stop working?**
   (Expected: `percentile_cont` — exact, Postgres sorts and interpolates,
   O(n log n) over the window. At high volume switch to a t-digest style
   approximation or pre-aggregated buckets, trading ~1% accuracy for O(1)
   memory. Bonus: why TTFT and total latency are separate columns.)

4. *(stretch)* **Why `Numeric` rather than `double precision` for cost?**
   (Expected: binary floating point cannot represent decimal fractions exactly,
   and the error compounds when summing thousands of sub-cent values. Also why
   8 decimal places: a cheap request costs a fraction of a cent and 4dp would
   round it to zero.)

---

## Phase 7 — Polish

### What I built

| Piece | Location |
|---|---|
| Integration test tier | [tests/integration/](tests/integration/) |
| RAG retrieval tests | [test_vector_search.py](tests/integration/test_vector_search.py) |
| Lua + aggregation SQL tests | [test_lua_and_sql.py](tests/integration/test_lua_and_sql.py) |
| README + mermaid architecture diagram | [README.md](README.md) |

### The gap this phase actually closed

Auditing the suite before writing anything turned up three things I had been
claiming were "verified" when they were only verified *by me, once, by hand*:

1. **RAG retrieval had no tests at all.** `test_chunking.py` covered splitting,
   but nothing touched `DocumentRepository.search` — the `<=>` operator, the
   ranking, the similarity floor. The one feature the phase brief named
   explicitly was the one with zero coverage.
2. **The rate limiter's Lua script was never executed by a test.** The unit
   tests ran a *Python re-implementation* of the token bucket. A
   re-implementation can be wrong in exactly the same way the real thing is, so
   those tests could never have caught a bug in the Lua.
3. **The metrics aggregation SQL had no coverage.** `percentile_cont` was
   asserted nowhere.

Phase 1's DECISIONS had promised "real-container integration tests come in
Phase 7". This delivers that rather than quietly dropping it.

### Key decisions

**1. Two tiers, with different jobs.** Unit tests (119) verify *our* logic with
fakes — fast, hermetic, runnable with no Docker. Integration tests (23) verify
what we handed to the database: the actual pgvector operator, the actual Lua,
the actual SQL. The split is deliberate; neither replaces the other.

**2. Every integration test runs in a rolled-back transaction.** The session is
bound to an *outer* transaction, so even service code that calls `commit()` —
all of them do — lands inside it and is discarded. Without this, running the
tier twice would accumulate rows in the developer's database. Verified: after a
full run, the 7 real request logs and 3 documents were untouched.

**3. The tier skips, it does not fail, when services are down.** `pytest` on a
laptop with no Docker stays green, with a message naming the command to fix it.
A test suite that fails for environmental reasons trains people to ignore red.

**4. The concurrency test is the point of the Lua test.** Firing 30 simultaneous
requests at a bucket of 5 and asserting *exactly* 5 pass is the assertion that
would fail if the refill/check/decrement were ever split into separate round
trips. It is the one test that actually justifies the design.

**5. SQL tests clear the table inside their own transaction.** Otherwise the
count assertions also see whatever real traffic the developer's database holds —
the test would pass on a fresh machine and fail on a used one. That is worse
than no test, because it teaches you to distrust the suite.

### Things the tests caught while being written

- **Event-loop scope mismatch.** The project pins a *session*-scoped fixture
  loop (Phase 1, so an engine can be shared), but a fixture holding a live
  asyncpg connection must run on the same loop as the test using it. Surfaced as
  `attached to a different loop`; fixed by pinning `loop_scope="function"` on
  the integration fixtures only.
- **A test racing its own network latency.** The refill test originally used
  100 tokens/sec — one token per 10ms — while its three setup round trips to
  Redis take longer than that. It was refilling a token during setup and passing
  or failing on timing luck. Slowed to 10 tokens/sec so the margin is real.
- **Two docstrings demoted to dead code** by an automated edit that inserted a
  statement above them. Caught on review, not by a tool.

### The README

Rewritten around a **mermaid architecture diagram** that shows the actual
dependency direction — middleware outermost-first, `routers → services →
repositories/providers`, and which components touch Postgres versus Redis. Plus
an ASCII flow for the RAG request path, because the ordering (embed once, use
the vector twice) is the non-obvious part.

`<i>` tags were removed from the mermaid labels: `<br/>` is supported by every
renderer, `<i>` is not, and renders as literal angle brackets on GitHub.

### Interview questions I should be able to answer

1. **You have unit tests with fakes and integration tests against real services.
   Why both — isn't one redundant?**
   (Expected: a fake can be wrong in the same way the code is, so a fake can
   never validate a contract you handed to someone else — the pgvector operator,
   the Lua script, the SQL. Conversely unit tests run anywhere in seconds and
   isolate *our* logic. Concrete example: the Python token bucket in the unit
   tests would agree with a broken Lua script.)

2. **How do your integration tests avoid polluting the database, and why does
   that matter more than it sounds?**
   (Expected: each test binds its session to an outer transaction that is rolled
   back, so even code calling `commit()` is contained. Without it the tier is
   not repeatable and the count assertions drift. Bonus: the SQL tests also
   clear the table *inside* that transaction, or they would count pre-existing
   rows and pass on a fresh machine while failing on a used one.)

3. **What single test justifies putting the rate-limit logic in Lua rather than
   Python?**
   (Expected: the 30-way concurrency test asserting exactly 5 of 30 simultaneous
   requests pass a bucket of 5. Split into separate GET/SET round trips, several
   interleave between read and write and more than 5 are admitted — silently,
   under exactly the load the limiter exists for.)

### Amendment — PDF ingestion

The upload route decoded every file with `raw.decode("utf-8", errors="replace")`.
For a PDF that "succeeds": the container's structure (`%PDF-1.4`,
`1 0 obj<</Type/Catalog…`) and its compressed streams as replacement characters
get chunked, embedded and stored as searchable content, and the upload returns
**201**. Verified by uploading one: the indexed text began `%PDF-1.4 / 1 0 obj…`.

Silently indexing garbage is worse than refusing the file, because nothing
signals that it happened.

**Detect by magic bytes, not `Content-Type`.** The declared type is supplied by
the client — browsers guess it from the extension, `curl -F` lets you assert
anything, and many clients send `application/octet-stream` for everything. The
first bytes are what the file actually is. A PDF renamed `.txt` now still works,
and a `.txt` that is really a PNG is still refused.

**`pypdf`, not pdfplumber/poppler.** Pure Python, so nothing new is installed in
the Docker image.

**Reject, with a diagnosis.** Known signatures (ZIP/docx, PNG, JPEG, legacy
Office, ELF…) produce a message naming the guess. Unknown binary is caught by a
NUL-byte check plus a replacement-character ratio.

**The ratio needed an absolute floor.** A pure 5% threshold rejects a
40-character note containing one mis-encoded quote — 2 bad chars in 38 is 5.3%.
So a handful of bad characters is allowed outright and the ratio only applies
beyond that. Found because a test written for the *intended* behaviour failed.

**Scanned PDFs raise rather than indexing empty.** A PDF with pages but no text
layer needs OCR; returning an empty document would look like success.

Verified end-to-end against a Flate-compressed PDF (the test fixture asserts the
text is *not* visible in the raw bytes, so a broken extractor cannot pass by
accident): indexed content is clean text with `[page N]` markers, and it
retrieves at 0.72 similarity. A PNG upload returns `415` with a useful message.
18 new tests.
