import { useState } from 'react'
import { Send, RotateCcw } from 'lucide-react'
import { api } from '../lib/api'
import ModelSelector from '../components/ModelSelector'
import PredictResult from '../components/PredictResult'
import ExplainCard from '../components/ExplainCard'

const SAMPLES = [
  "WINNER!! You've been selected for a £1000 cash prize. Call 08061701461 now!",
  "Hey, are we still meeting for lunch tomorrow at 1pm?",
  "FREE entry in 2 a weekly comp to win FA Cup final tkts. Text FA to 87121",
  "Can you pick up milk on your way home? Thanks!",
]

export default function PredictPage() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('bilstm')
  const [result, setResult] = useState(null)
  const [explain, setExplain] = useState(null)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState(null)

  async function handlePredict() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setExplain(null)
    try {
      const res = await api.predict(text.trim(), model)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleExplain() {
    if (!result) return
    setExplaining(true)
    try {
      const res = await api.explain(result.text, result.label, result.probability, result.model)
      setExplain(res)
    } catch (e) {
      setExplain({ error: e.message })
    } finally {
      setExplaining(false)
    }
  }

  function handleReset() {
    setText(''); setResult(null); setExplain(null); setError(null)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <p className="text-xs font-mono text-accent uppercase tracking-widest mb-1">Single Message</p>
        <h1 className="font-display text-3xl font-bold text-text">Spam Detector</h1>
        <p className="text-text-dim text-sm mt-1">Analyze any SMS message for spam signals using ML models.</p>
      </div>

      {/* Input card */}
      <div className="card space-y-4">
        <div>
          <label className="label">Message</label>
          <textarea
            className="textarea"
            rows={4}
            placeholder="Paste an SMS message here..."
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && e.ctrlKey && handlePredict()}
          />
          <p className="text-xs text-muted mt-1.5 font-mono">{text.length} chars · Ctrl+Enter to submit</p>
        </div>

        {/* Samples */}
        <div>
          <label className="label">Try a sample</label>
          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((s, i) => (
              <button
                key={i}
                onClick={() => setText(s)}
                className="text-xs font-mono text-text-dim bg-bg border border-border px-3 py-1.5 rounded-lg hover:border-accent/50 hover:text-accent transition-colors text-left truncate max-w-xs"
              >
                {s.slice(0, 40)}…
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3 pt-1">
          <div className="w-44">
            <label className="label">Model</label>
            <ModelSelector value={model} onChange={setModel} />
          </div>
          <div className="flex gap-2 mt-5">
            <button onClick={handlePredict} disabled={loading || !text.trim()} className="btn-primary flex items-center gap-2">
              <Send size={14} />
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
            {result && (
              <button onClick={handleReset} className="btn-ghost flex items-center gap-2">
                <RotateCcw size={14} /> Reset
              </button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="card border-red-500/20 text-danger text-sm font-mono">{error}</div>
      )}

      {result && (
        <PredictResult result={result} onExplain={handleExplain} explaining={explaining} />
      )}

      {explain && <ExplainCard data={explain} />}
    </div>
  )
}
