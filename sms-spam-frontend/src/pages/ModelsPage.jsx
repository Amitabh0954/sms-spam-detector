import { useEffect, useState } from 'react'
import { BarChart3, Cpu, TrendingUp } from 'lucide-react'
import { api } from '../lib/api'

const MODEL_LABELS = {
  mnb_tfidf: 'MNB · TF-IDF',
  mnb_count: 'MNB · Count',
  simplernn: 'Simple RNN',
  lstm: 'LSTM',
  gru: 'GRU',
  bilstm: 'BiLSTM',
}

const METRIC_COLORS = {
  accuracy:  '#4f9cf9',
  f1:        '#4ff994',
  precision: '#f9c74f',
  recall:    '#f94f4f',
  roc_auc:   '#c084fc',
}

function MetricBar({ value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 prob-bar">
        <div className="prob-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{pct}%</span>
    </div>
  )
}

export default function ModelsPage() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [sort, setSort] = useState('f1')

  useEffect(() => {
    api.models()
      .then(ms => setModels(ms))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const available = models.filter(m => m.available)
  const sorted = [...available].sort((a, b) => (b.metrics?.[sort] || 0) - (a.metrics?.[sort] || 0))
  const best = sorted[0]

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-mono text-accent uppercase tracking-widest mb-1">Model Registry</p>
        <h1 className="font-display text-3xl font-bold text-text">Compare Models</h1>
        <p className="text-text-dim text-sm mt-1">Performance metrics across all loaded models.</p>
      </div>

      {loading ? (
        <div className="card text-center text-muted font-mono text-sm py-12">Loading models...</div>
      ) : (
        <>
          {/* Sort control */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-muted">Sort by:</span>
            {['accuracy', 'f1', 'precision', 'recall', 'roc_auc'].map(m => (
              <button
                key={m}
                onClick={() => setSort(m)}
                className={`text-xs font-mono px-3 py-1.5 rounded-lg border transition-colors ${
                  sort === m
                    ? 'border-accent/40 bg-accent/10 text-accent'
                    : 'border-border text-muted hover:text-text-dim'
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Best model highlight */}
          {best && (
            <div className="card border-accent/20 glow-accent">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp size={14} className="text-accent" />
                <span className="text-xs font-mono text-accent uppercase tracking-widest">Best by {sort}</span>
              </div>
              <div className="font-display text-2xl font-bold text-text">{MODEL_LABELS[best.name] || best.name}</div>
              <div className="text-text-dim text-sm mt-1 font-mono">
                {sort}: {Math.round((best.metrics?.[sort] || 0) * 100)}%
                · threshold: {best.threshold ? Math.round(best.threshold * 100) + '%' : 'N/A'}
              </div>
            </div>
          )}

          {/* Model cards grid */}
          <div className="grid grid-cols-1 gap-4">
            {sorted.map((m, idx) => (
              <div
                key={m.name}
                className={`card transition-all ${idx === 0 ? 'border-accent/30' : ''}`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-mono font-bold
                      ${m.type === 'sklearn' ? 'bg-warn/10 text-warn border border-warn/20' : 'bg-accent/10 text-accent border border-accent/20'}`}>
                      {idx + 1}
                    </div>
                    <div>
                      <div className="font-display font-semibold text-text">{MODEL_LABELS[m.name] || m.name}</div>
                      <div className="text-xs font-mono text-muted">{m.type} · threshold {m.threshold ? Math.round(m.threshold * 100) + '%' : 'N/A'}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-display font-bold text-accent">
                      {Math.round((m.metrics?.f1 || 0) * 100)}%
                    </div>
                    <div className="text-xs font-mono text-muted">F1 Score</div>
                  </div>
                </div>

                {m.metrics && (
                  <div className="space-y-2.5">
                    {['accuracy', 'f1', 'precision', 'recall', 'roc_auc'].map(metric => (
                      <div key={metric}>
                        <div className="flex justify-between mb-1">
                          <span className="text-xs font-mono text-muted capitalize">{metric}</span>
                        </div>
                        <MetricBar value={m.metrics[metric]} color={METRIC_COLORS[metric]} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Unavailable models */}
          {models.filter(m => !m.available).length > 0 && (
            <div className="card border-dashed">
              <p className="text-xs font-mono text-muted mb-2 uppercase tracking-widest">Not Loaded</p>
              <div className="flex gap-2 flex-wrap">
                {models.filter(m => !m.available).map(m => (
                  <span key={m.name} className="text-xs font-mono text-muted bg-bg border border-border px-3 py-1.5 rounded-lg">
                    {MODEL_LABELS[m.name] || m.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
