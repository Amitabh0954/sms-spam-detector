import { useState } from 'react'
import { Brain, Shield, Search, AlertOctagon } from 'lucide-react'
import { api } from '../lib/api'
import ModelSelector from '../components/ModelSelector'
import ExplainCard from '../components/ExplainCard'

const TABS = [
  { id: 'explain',  label: 'Explain',       icon: Brain,        desc: 'Explain a prediction' },
  { id: 'evasion',  label: 'Evasion Check',  icon: Shield,       desc: 'Detect spam evasion tactics' },
]

export default function ExplainPage() {
  const [tab, setTab] = useState('explain')
  const [text, setText] = useState('')
  const [model, setModel] = useState('bilstm')
  const [label, setLabel] = useState('spam')
  const [probability, setProbability] = useState(0.85)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      let res
      if (tab === 'explain') {
        res = await api.explain(text.trim(), label, parseFloat(probability), model)
      } else {
        res = await api.rewriteCheck(text.trim())
      }
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-mono text-accent uppercase tracking-widest mb-1">AI Intelligence</p>
        <h1 className="font-display text-3xl font-bold text-text">Explainability</h1>
        <p className="text-text-dim text-sm mt-1">Powered by Groq · LLaMA 3.3 70B</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => { setTab(t.id); setResult(null); setError(null) }}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm transition-all ${
              tab === t.id
                ? 'bg-accent/10 border-accent/30 text-accent'
                : 'border-border text-text-dim hover:text-text hover:border-border'
            }`}
          >
            <t.icon size={14} />
            {t.label}
          </button>
        ))}
      </div>

      {/* Input card */}
      <div className="card space-y-4">
        <div>
          <label className="label">Message</label>
          <textarea
            className="textarea"
            rows={4}
            placeholder={
              tab === 'explain'
                ? 'Enter the message to explain...'
                : 'Enter a suspicious message to check for evasion tactics...'
            }
            value={text}
            onChange={e => setText(e.target.value)}
          />
        </div>

        {tab === 'explain' && (
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="label">Model</label>
              <ModelSelector value={model} onChange={setModel} />
            </div>
            <div>
              <label className="label">Predicted Label</label>
              <select
                className="input"
                value={label}
                onChange={e => setLabel(e.target.value)}
              >
                <option value="spam">Spam</option>
                <option value="ham">Ham</option>
              </select>
            </div>
            <div>
              <label className="label">Probability (0–1)</label>
              <input
                type="number"
                min={0} max={1} step={0.01}
                className="input font-mono"
                value={probability}
                onChange={e => setProbability(e.target.value)}
              />
            </div>
          </div>
        )}

        {tab === 'evasion' && (
          <div className="bg-warn/5 border border-warn/20 rounded-lg p-3 flex items-start gap-2">
            <AlertOctagon size={14} className="text-warn mt-0.5 shrink-0" />
            <p className="text-warn/80 text-xs leading-relaxed">
              Evasion check detects obfuscation techniques spammers use to bypass filters — misspellings, Unicode tricks, unusual spacing, and more.
            </p>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
          className="btn-primary flex items-center gap-2"
        >
          <Brain size={14} />
          {loading ? 'Analyzing with Groq...' : tab === 'explain' ? 'Explain Prediction' : 'Check for Evasion'}
        </button>
      </div>

      {error && <div className="card border-red-500/20 text-danger text-sm font-mono">{error}</div>}

      {/* Explain result */}
      {result && tab === 'explain' && <ExplainCard data={result} />}

      {/* Evasion result */}
      {result && tab === 'evasion' && !result.error && (
        <div className="card animate-slide-up space-y-4">
          <div className="flex items-center gap-3 mb-2">
            <Shield size={16} className={result.evasion_risk === 'High' ? 'text-danger' : result.evasion_risk === 'Medium' ? 'text-warn' : 'text-success'} />
            <span className="font-display font-semibold">Evasion Analysis</span>
            <span className={`ml-auto text-xs font-mono px-2 py-0.5 rounded-full border ${
              result.evasion_risk === 'High' ? 'bg-red-500/10 text-danger border-red-500/20' :
              result.evasion_risk === 'Medium' ? 'bg-warn/10 text-warn border-warn/20' :
              'bg-green-500/10 text-success border-green-500/20'
            }`}>
              {result.evasion_risk} Risk
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-bg rounded-lg p-3 border border-border">
              <span className="label">Likely Spam</span>
              <div className={`font-display font-bold text-lg ${result.is_likely_spam ? 'text-danger' : 'text-success'}`}>
                {result.is_likely_spam ? 'Yes' : 'No'}
              </div>
            </div>
            <div className="bg-bg rounded-lg p-3 border border-border">
              <span className="label">Action</span>
              <div className="text-text text-sm font-mono">{result.recommended_action || 'N/A'}</div>
            </div>
          </div>

          {result.evasion_techniques?.length > 0 && (
            <div>
              <span className="label">Evasion Techniques</span>
              <div className="flex flex-wrap gap-2">
                {result.evasion_techniques.map((t, i) => (
                  <span key={i} className="badge-spam">{t}</span>
                ))}
              </div>
            </div>
          )}

          {result.red_flags?.length > 0 && (
            <div>
              <span className="label">Red Flags</span>
              <ul className="space-y-1">
                {result.red_flags.map((f, i) => (
                  <li key={i} className="text-sm text-text-dim flex items-start gap-2">
                    <span className="text-danger font-mono text-xs mt-0.5">!</span>{f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.reasoning && (
            <div className="bg-bg rounded-lg p-3 border border-border">
              <span className="label">Reasoning</span>
              <p className="text-text-dim text-sm leading-relaxed">{result.reasoning}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
