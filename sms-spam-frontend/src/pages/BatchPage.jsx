import { useState } from 'react'
import { LayoutList, Play, Download, Zap } from 'lucide-react'
import { api } from '../lib/api'
import ModelSelector from '../components/ModelSelector'

const PLACEHOLDER = `WINNER!! You've been selected for a cash prize. Call now!
Hey, are we still meeting for lunch tomorrow?
FREE entry in 2 a weekly comp to win FA Cup final tkts
Can you pick up milk on your way home?
Congratulations! ur awarded a £500 prize. Reply WIN to claim`

export default function BatchPage() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('bilstm')
  const [results, setResults] = useState([])
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)

  const lines = text.split('\n').map(l => l.trim()).filter(Boolean)

  async function handleBatch() {
    if (!lines.length) return
    setLoading(true)
    setError(null)
    setResults([])
    setAnalysis(null)
    try {
      const res = await api.predictBatch(lines, model)
      setResults(res.results)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleAnalyze() {
    if (!results.length) return
    setAnalyzing(true)
    try {
      const res = await api.analyze(results, model)
      setAnalysis(res)
    } catch (e) {
      setAnalysis({ error: e.message })
    } finally {
      setAnalyzing(false)
    }
  }

  function exportCSV() {
    const header = 'text,label,probability,confidence,threshold'
    const rows = results.map(r =>
      `"${r.text.replace(/"/g, '""')}",${r.label},${r.probability},${r.confidence},${r.threshold}`
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'spam_results.csv'
    a.click()
  }

  const spamCount = results.filter(r => r.label === 'spam').length
  const hamCount = results.filter(r => r.label === 'ham').length

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-mono text-accent uppercase tracking-widest mb-1">Batch Analysis</p>
        <h1 className="font-display text-3xl font-bold text-text">Multi-Message Scan</h1>
        <p className="text-text-dim text-sm mt-1">Paste multiple messages (one per line) to analyze in bulk.</p>
      </div>

      <div className="card space-y-4">
        <div>
          <label className="label">Messages (one per line)</label>
          <textarea
            className="textarea"
            rows={8}
            placeholder={PLACEHOLDER}
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <p className="text-xs text-muted mt-1.5 font-mono">{lines.length} messages</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="w-44">
            <label className="label">Model</label>
            <ModelSelector value={model} onChange={setModel} />
          </div>
          <button onClick={handleBatch} disabled={loading || !lines.length} className="btn-primary flex items-center gap-2 mt-5">
            <Play size={14} />
            {loading ? 'Scanning...' : `Scan ${lines.length || ''} Messages`}
          </button>
        </div>
      </div>

      {error && <div className="card border-red-500/20 text-danger text-sm font-mono">{error}</div>}

      {results.length > 0 && (
        <div className="space-y-4 animate-slide-up">
          {/* Summary bar */}
          <div className="card">
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-bg rounded-lg p-3 border border-border text-center">
                <div className="text-2xl font-display font-bold text-text">{results.length}</div>
                <div className="text-xs font-mono text-muted">Total</div>
              </div>
              <div className="bg-bg rounded-lg p-3 border border-red-500/20 text-center glow-danger">
                <div className="text-2xl font-display font-bold text-danger">{spamCount}</div>
                <div className="text-xs font-mono text-muted">Spam</div>
              </div>
              <div className="bg-bg rounded-lg p-3 border border-green-500/20 text-center glow-success">
                <div className="text-2xl font-display font-bold text-success">{hamCount}</div>
                <div className="text-xs font-mono text-muted">Ham</div>
              </div>
            </div>
            <div className="prob-bar mb-3">
              <div
                className="prob-bar-fill"
                style={{ width: `${(spamCount / results.length) * 100}%`, background: 'linear-gradient(90deg, #f9c74f, #f94f4f)' }}
              />
            </div>
            <div className="flex justify-between">
              <button onClick={handleAnalyze} disabled={analyzing} className="btn-primary flex items-center gap-2 text-xs">
                <Zap size={13} />
                {analyzing ? 'AI Analyzing...' : 'AI Batch Analysis'}
              </button>
              <button onClick={exportCSV} className="btn-ghost flex items-center gap-2 text-xs">
                <Download size={13} /> Export CSV
              </button>
            </div>
          </div>

          {/* Results table */}
          <div className="card p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left px-4 py-3 text-xs font-mono text-muted uppercase tracking-wider">#</th>
                    <th className="text-left px-4 py-3 text-xs font-mono text-muted uppercase tracking-wider">Message</th>
                    <th className="text-left px-4 py-3 text-xs font-mono text-muted uppercase tracking-wider">Label</th>
                    <th className="text-left px-4 py-3 text-xs font-mono text-muted uppercase tracking-wider">Prob</th>
                    <th className="text-left px-4 py-3 text-xs font-mono text-muted uppercase tracking-wider">Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-border/20 transition-colors">
                      <td className="px-4 py-3 text-muted font-mono text-xs">{i + 1}</td>
                      <td className="px-4 py-3 max-w-xs">
                        <p className="truncate text-text-dim text-xs" title={r.text}>{r.text}</p>
                      </td>
                      <td className="px-4 py-3">
                        <span className={r.label === 'spam' ? 'badge-spam' : 'badge-ham'}>
                          {r.label}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-text-dim">{Math.round(r.probability * 100)}%</td>
                      <td className="px-4 py-3 font-mono text-xs">
                        <span className={r.label === 'spam' ? 'text-danger' : 'text-success'}>
                          {Math.round(r.confidence * 100)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* AI Analysis */}
          {analysis && !analysis.error && (
            <div className="card animate-slide-up space-y-4">
              <div className="flex items-center gap-2">
                <Zap size={16} className="text-warn" />
                <span className="font-display font-semibold">AI Batch Analysis</span>
                {analysis.threat_level && (
                  <span className="ml-auto text-xs font-mono px-2 py-0.5 rounded-full bg-red-500/10 text-danger border border-red-500/20">
                    {analysis.threat_level}
                  </span>
                )}
              </div>
              {analysis.summary && (
                <p className="text-text-dim text-sm leading-relaxed">{analysis.summary}</p>
              )}
              {analysis.top_patterns?.length > 0 && (
                <div>
                  <span className="label">Top Patterns</span>
                  <ul className="space-y-1">
                    {analysis.top_patterns.map((p, i) => (
                      <li key={i} className="text-sm text-text-dim flex items-start gap-2">
                        <span className="text-accent font-mono text-xs mt-0.5">→</span>{p}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {analysis.recommendation && (
                <div className="bg-warn/5 border border-warn/20 rounded-lg p-3 text-warn/80 text-xs">
                  {analysis.recommendation}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
