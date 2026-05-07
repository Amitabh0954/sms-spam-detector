const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  predict: (text, model = 'bilstm') =>
    request('/predict', { method: 'POST', body: JSON.stringify({ text, model }) }),

  predictBatch: (texts, model = 'bilstm') =>
    request('/predict/batch', { method: 'POST', body: JSON.stringify({ texts, model }) }),

  explain: (text, label, probability, model = 'bilstm') =>
    request('/intelligence/explain', {
      method: 'POST',
      body: JSON.stringify({ text, label, probability, model }),
    }),

  analyze: (results, model = 'bilstm') =>
    request('/intelligence/analyze', {
      method: 'POST',
      body: JSON.stringify({ results, model }),
    }),

  rewriteCheck: (text) =>
    request('/intelligence/rewrite-check', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),

  models: () => request('/models'),

  health: () => request('/health'),
}
