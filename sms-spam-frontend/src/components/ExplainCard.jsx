import { Lightbulb, Tag, ShieldAlert, Info } from 'lucide-react'

export default function ExplainCard({ data }) {
  if (!data) return null
  if (data.error) return (
    <div className="card border-red-500/20 text-danger text-sm font-mono">{data.error}</div>
  )

  return (
    <div className="card animate-slide-up space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Lightbulb size={16} className="text-warn" />
        <span className="font-display font-semibold text-text">AI Explanation</span>
        {data.confidence_level && (
          <span className="ml-auto text-xs font-mono text-text-dim">{data.confidence_level}</span>
        )}
      </div>

      {data.verdict && (
        <div className="bg-bg rounded-lg p-3 border border-border">
          <span className="label">Verdict</span>
          <p className="text-text text-sm">{data.verdict}</p>
        </div>
      )}

      {data.reasoning && (
        <div className="bg-bg rounded-lg p-3 border border-border">
          <span className="label">Reasoning</span>
          <p className="text-text-dim text-sm leading-relaxed">{data.reasoning}</p>
        </div>
      )}

      {data.key_signals?.length > 0 && (
        <div>
          <span className="label flex items-center gap-1.5"><Tag size={11} />Key Signals</span>
          <div className="flex flex-wrap gap-2 mt-1.5">
            {data.key_signals.map((s, i) => (
              <span key={i} className="bg-accent/10 text-accent border border-accent/20 text-xs px-2.5 py-1 rounded-full font-mono">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {data.spam_category && data.spam_category !== 'None' && (
        <div className="bg-bg rounded-lg p-3 border border-border flex items-start gap-2">
          <ShieldAlert size={14} className="text-danger mt-0.5 shrink-0" />
          <div>
            <span className="label">Category</span>
            <p className="text-danger text-sm font-mono">{data.spam_category}</p>
          </div>
        </div>
      )}

      {data.tip && (
        <div className="bg-warn/5 border border-warn/20 rounded-lg p-3 flex items-start gap-2">
          <Info size={14} className="text-warn mt-0.5 shrink-0" />
          <p className="text-warn/80 text-xs leading-relaxed">{data.tip}</p>
        </div>
      )}
    </div>
  )
}
