import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { ChevronDown } from 'lucide-react'

const MODEL_TYPES = {
  mnb_tfidf: 'MNB · TF-IDF',
  mnb_count: 'MNB · Count',
  simplernn: 'SimpleRNN',
  lstm: 'LSTM',
  gru: 'GRU',
  bilstm: 'BiLSTM',
}

export default function ModelSelector({ value, onChange }) {
  const [models, setModels] = useState([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    api.models().then(ms => setModels(ms.filter(m => m.available))).catch(() => {})
  }, [])

  const selected = models.find(m => m.name === value)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="input flex items-center justify-between"
      >
        <span className="font-mono text-accent">{MODEL_TYPES[value] || value}</span>
        <ChevronDown size={14} className={`text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-surface border border-border rounded-lg overflow-hidden z-20 shadow-xl animate-fade-in">
          {models.map(m => (
            <button
              key={m.name}
              onClick={() => { onChange(m.name); setOpen(false) }}
              className={`w-full px-4 py-2.5 flex items-center justify-between hover:bg-border/50 transition-colors text-left ${m.name === value ? 'text-accent' : 'text-text-dim'}`}
            >
              <span className="text-sm font-mono">{MODEL_TYPES[m.name] || m.name}</span>
              {m.metrics?.f1 && (
                <span className="text-xs text-muted">F1: {m.metrics.f1}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
