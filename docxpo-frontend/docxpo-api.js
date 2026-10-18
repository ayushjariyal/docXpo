// docXpo console — API client with a mock fallback.
// Live mode talks to the same-origin FastAPI app (/v1/*). If no docXpo server
// answers at this origin, the console switches to mock data so it stays usable.
(function () {
  const sleep = (ms, signal) => new Promise((res, rej) => {
    const t = setTimeout(res, ms);
    if (signal) signal.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('Aborted', 'AbortError')); }, { once: true });
  });

  function compact(n) {
    if (n === null || n === undefined) return '—';
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
    return Number(n).toLocaleString();
  }
  function money(v) {
    const n = Number(v);
    if (!isFinite(n) || n === 0) return '$0';
    if (n < 0.01) return '$' + n.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
    if (n < 1000) return '$' + n.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    return '$' + compact(n);
  }
  const ms = (v) => (v === null || v === undefined ? '—' : Math.round(v).toLocaleString() + 'ms');

  async function errorDetail(res) {
    const raw = await res.text();
    try { const p = JSON.parse(raw); return p.detail ?? p; } catch { return raw; }
  }
  function limitMessage(res, detail) {
    if (res.status === 401) return 'Unauthorized — set your API key in settings.';
    if (res.status === 429 && detail && detail.type === 'RateLimitExceeded') {
      const retry = res.headers.get('Retry-After');
      return `Rate limited by docXpo${retry ? ` — retry in ${retry}s` : ''}. ${res.headers.get('X-RateLimit-Limit') ?? '?'}/min allowed.`;
    }
    return typeof detail === 'string' ? detail : (detail?.error ?? JSON.stringify(detail));
  }

  async function probe() {
    try {
      const r = await fetch('/v1/documents?limit=1', { headers: { accept: 'application/json' } });
      if (r.status === 401) return true;
      if (!r.ok) return false;
      return (r.headers.get('content-type') || '').includes('json');
    } catch { return false; }
  }

  // fetch + manual SSE parsing: EventSource can't POST a JSON body.
  async function streamSSE(url, payload, headers, signal, onEvent) {
    const res = await fetch(url, { method: 'POST', headers: { ...headers, 'content-type': 'application/json' }, body: JSON.stringify(payload), signal });
    if (!res.ok) { const e = new Error(limitMessage(res, await errorDetail(res))); e.http = true; throw e; }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, i); buf = buf.slice(i + 2);
        let event = 'message', data = '';
        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7);
          else if (line.startsWith('data: ')) data += line.slice(6);
        }
        if (data) onEvent(event, JSON.parse(data));
      }
    }
  }

  // ---- mock data -----------------------------------------------------------
  const DEFAULT_MODELS = { gemini: 'gemini-flash-latest', anthropic: 'claude-sonnet-4-6', openai: 'gpt-4o-mini', ollama: 'llama3.2' };
  const now = Date.now();
  const MOCK_DOCS = [
    { id: 'd1', filename: 'DECISIONS.md', content_type: 'text/markdown', char_count: 61806, chunk_count: 52, created_at: new Date(now - 864e5 * 3).toISOString() },
    { id: 'd2', filename: 'architecture-notes.pdf', content_type: 'application/pdf', char_count: 21440, chunk_count: 18, created_at: new Date(now - 864e5 * 2).toISOString() },
    { id: 'd3', filename: 'rate-limit-runbook.md', content_type: 'text/markdown', char_count: 7920, chunk_count: 7, created_at: new Date(now - 864e5).toISOString() },
    { id: 'd4', filename: 'pricing-2026.csv', content_type: 'text/csv', char_count: 2310, chunk_count: 3, created_at: new Date(now - 36e5 * 5).toISOString() },
    { id: 'd5', filename: 'pasted.txt', content_type: 'text/plain', char_count: 640, chunk_count: 1, created_at: new Date(now - 36e5).toISOString() },
  ];
  const SOURCE_POOL = [
    { document_id: 'd1', filename: 'DECISIONS.md', chunk_index: 14, similarity: 0.8731, excerpt: 'Rate limiting is a token bucket per API key, held in Redis. The check and the decrement run inside one Lua script, so two concurrent requests can never both spend the last token.' },
    { document_id: 'd3', filename: 'rate-limit-runbook.md', chunk_index: 2, similarity: 0.8412, excerpt: 'Defaults: 60 requests per minute sustained with a burst of 20. Every response carries X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset so clients can back off before they are rejected.' },
    { document_id: 'd1', filename: 'DECISIONS.md', chunk_index: 15, similarity: 0.7968, excerpt: 'When the bucket is empty the middleware returns 429 with a Retry-After header. /health is exempt so orchestrators can always probe the service.' },
    { document_id: 'd2', filename: 'architecture-notes.pdf', chunk_index: 6, similarity: 0.662, excerpt: 'With API_KEYS unset the service runs in open mode and meters callers by IP address instead of by key fingerprint.' },
    { document_id: 'd2', filename: 'architecture-notes.pdf', chunk_index: 9, similarity: 0.5214, excerpt: 'Keys are fingerprinted before they are used as bucket identifiers; the raw key is never written to Redis or to the request log.' },
  ];
  const RAG_ANSWER = 'docXpo limits requests per API key with a token bucket stored in Redis [1]. Each key gets 60 requests per minute sustained and a burst of 20, and every response reports the remaining budget in X-RateLimit-* headers [2].\n\nThe check-and-decrement runs as a single Lua script, so concurrent requests can\u2019t both take the last token [1]. Once the bucket is empty the gateway answers 429 with Retry-After, and /health is exempt [3]. If API_KEYS is unset, callers are metered by IP instead [4].';
  const CHAT_RAG = 'Retrieval-augmented generation answers a question by first searching a document store for the passages most similar to it, then handing those passages to the model as context. The model writes its answer from that context, so it can cite sources it was never trained on.';
  const CHAT_CACHE = 'A semantic cache stores answers keyed by the embedding of the question rather than its exact text. A new question whose embedding is close enough to a stored one returns the stored answer immediately, spending no generation tokens.';
  const CHAT_MOCK = 'This reply is simulated: no docXpo server answered at this origin, so the console is running on mock data. Start the stack with docker compose up -d --build and reload to stream from your configured provider.';

  const norm = (q) => q.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
  const pickSources = (k = 5, min = 0.5) => SOURCE_POOL.filter((s) => s.similarity >= min).slice(0, k).map((s, i) => ({ ...s, n: i + 1 }));

  async function mockStream(kind, payload, signal, onEvent, ctx) {
    const q = kind === 'rag' ? payload.question : (payload.messages.filter((m) => m.role === 'user').slice(-1)[0] || {}).content || '';
    const key = norm(q);
    const hit = kind === 'rag' && (ctx.cache[key] || Object.values(ctx.cache).find((c) => c.alias && c.alias.includes(key)));
    if (hit) {
      await sleep(40, signal);
      onEvent('sources', { sources: hit.sources });
      onEvent('token', { text: hit.answer });
      onEvent('done', { cached: true, cache_similarity: ctx.cache[key] ? 1 : 0.9912, latency_ms: 1.4, tokens_saved: hit.tokens, cache_age_seconds: (Date.now() - hit.t) / 1000, model: hit.model, cached_question: hit.q, usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } });
      return;
    }
    const provider = payload.provider || 'gemini';
    const model = payload.model || DEFAULT_MODELS[provider] || 'gemini-flash-latest';
    const t0 = performance.now();
    await sleep(320 + Math.random() * 260, signal);
    let answer, sources = [], finish = 'stop';
    if (kind === 'rag') {
      sources = pickSources(payload.top_k, payload.min_similarity);
      onEvent('sources', { sources });
      if (!sources.length) { answer = 'Nothing in the indexed corpus cleared min_sim for this question, so there is no context to answer from.'; finish = 'no_context'; }
      else answer = RAG_ANSWER.replace(/\[(\d+)\]/g, (m, n) => (Number(n) <= sources.length ? m : ''));
    } else {
      const l = q.toLowerCase();
      answer = /rag|retriev/.test(l) ? CHAT_RAG : /cache/.test(l) ? CHAT_CACHE : CHAT_MOCK;
    }
    const pieces = answer.split(/(\s+)/);
    for (let i = 0; i < pieces.length; i += 4) {
      await sleep(26, signal);
      onEvent('token', { text: pieces.slice(i, i + 4).join('') });
    }
    const out = Math.round(answer.length / 4.2);
    const inp = kind === 'rag' ? 380 + sources.length * 64 : 14 + payload.messages.reduce((a, m) => a + Math.round(m.content.length / 4), 0);
    const latency = performance.now() - t0;
    if (kind === 'rag' && sources.length) ctx.cache[key] = { q, answer, sources, model, tokens: inp + out, t: Date.now() };
    onEvent('done', { provider, model, usage: { input_tokens: inp, output_tokens: out, total_tokens: inp + out }, finish_reason: finish, latency_ms: latency, cached: false });
  }

  function mockRetrieve(question, k, min) { return { question, sources: pickSources(k, min) }; }

  function seed(nextId) {
    const t = Date.now();
    const srcs = pickSources(5, 0.5);
    const q1 = 'How does docXpo rate-limit requests?';
    const a1 = nextId(), a2 = nextId(), a3 = nextId(), a4 = nextId(), c1 = nextId(), c2 = nextId();
    const cache = { [norm(q1)]: { q: q1, answer: RAG_ANSWER, sources: srcs, model: 'gemini-flash-latest', tokens: 1032, t: t - 194e3, alias: [norm('How are requests rate limited?')] } };
    return {
      cache, chatSel: c2, ragSel: a2,
      chat: [
        { id: c1, role: 'user', text: 'What is retrieval-augmented generation, in two sentences?' },
        { id: c2, role: 'assistant', text: CHAT_RAG, receipt: { cached: false, provider: 'gemini', model: 'gemini-flash-latest', input: 16, output: 62, ttft: 412, total: 842, finish: 'stop' }, sources: [] },
      ],
      rag: [
        { id: a1, role: 'user', text: q1 },
        { id: a2, role: 'assistant', text: RAG_ANSWER, sources: srcs, receipt: { cached: false, provider: 'gemini', model: 'gemini-flash-latest', input: 700, output: 332, ttft: 604, total: 2486, finish: 'stop' } },
        { id: a3, role: 'user', text: 'How are requests rate limited?' },
        { id: a4, role: 'assistant', text: RAG_ANSWER, sources: srcs, receipt: { cached: true, similarity: 0.9912, latency: 1.4, saved: 1032, age: 194, model: 'gemini-flash-latest', cachedQuestion: q1 } },
      ],
    };
  }

  function mockMetrics(hours) {
    const f = { 1: 0.08, 24: 1, 168: 4.6, 0: 9.3 }[Number(hours)] ?? 1;
    const r = (n) => Math.max(0, Math.round(n * f));
    const by = [
      { provider: 'gemini', model: 'gemini-flash-latest', requests: r(96), cache_hits: r(31), cost_usd: 0.0187 * f, p95_latency_ms: 2210 },
      { provider: 'anthropic', model: 'claude-sonnet-4-6', requests: r(21), cache_hits: r(6), cost_usd: 0.0214 * f, p95_latency_ms: 3120 },
      { provider: 'ollama', model: 'llama3.2', requests: r(11), cache_hits: 0, cost_usd: 0, p95_latency_ms: 4020 },
    ];
    const requests = by.reduce((a, m) => a + m.requests, 0);
    const hits = by.reduce((a, m) => a + m.cache_hits, 0);
    return {
      summary: {
        requests, errors: r(2), cache_hits: hits, cache_hit_rate: requests ? hits / requests : 0,
        tokens: { total: r(214380) },
        cost_usd: { spent: 0.0401 * f, saved_by_cache: 0.0117 * f, unpriced_requests: 0 },
        latency_ms: { p50: 842, p95: 2480, p99: 3910 },
      },
      by_model: by,
    };
  }
  function mockRecent() {
    const rows = [
      ['/v1/rag/query', 'gemini-flash-latest', true, 0, 0, 1.4, 0],
      ['/v1/rag/query', 'gemini-flash-latest', false, 700, 332, 2486, 0.000152],
      ['/v1/chat', 'gemini-flash-latest', false, 16, 62, 842, 0.000021],
      ['/v1/rag/retrieve', 'text-embedding-004', false, 9, 0, 118, 0.000001],
      ['/v1/chat', 'claude-sonnet-4-6', false, 412, 588, 3120, 0.010056],
      ['/v1/rag/query', 'claude-sonnet-4-6', false, 905, 410, 2904, 0.008865],
      ['/v1/chat', 'llama3.2', false, 38, 214, 4020, 0],
      ['/v1/rag/query', 'gemini-flash-latest', true, 0, 0, 1.1, 0],
    ];
    return rows.map((x, i) => ({ created_at: new Date(Date.now() - (i * 7 + 2) * 6e4).toISOString(), endpoint: x[0], model: x[1], cached: x[2], input_tokens: x[3], output_tokens: x[4], latency_ms: x[5], cost_usd: x[6] }));
  }

  window.DocxpoAPI = {
    compact, money, ms, errorDetail, limitMessage, probe, streamSSE, sleep,
    mockStream, mockRetrieve, mockMetrics, mockRecent, seed, DEFAULT_MODELS,
    mockDocs: () => MOCK_DOCS.map((d) => ({ ...d })),
  };
})();
