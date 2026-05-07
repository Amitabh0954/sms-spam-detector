import { AlertTriangle, CheckCircle, Zap, Clock } from 'lucide-react'

export default function PredictResult({ result, onExplain, explaining }) {
  if (!result) return null

  const isSpam = result.label === 'spam'
  const confPct = Math.round(result.confidence * 100)
  const probPct = Math.round(result.probability * 100)

  return (
    <div className={`card animate-slide-up ${isSpam ? 'glow-danger border-red-500/20' : 'glow-success border-green-500/20'}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          {isSpam
            ? <AlertTriangle size={20} className="text-danger" />
            : <CheckCircle size={20} className="text-success" />
          }
          <div>
            <div className="font-display font-bold text-lg">
              {isSpam ? 'Spam Detected' : 'Looks Legitimate'}
            </div>
            <div className="text-xs font-mono text-text-dim">via {result.model}</div>
          </div>
        </div>
        <span className={isSpam ? 'badge-spam' : 'badge-ham'}>
          {result.label.toUpperCase()}
        </span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-5">
        <div className="bg-bg rounded-lg p-3 border border-border">
          <span className="label">Confidence</span>
          <div className={`text-2xl font-display font-bold ${isSpam ? 'text-danger' : 'text-success'}`}>
            {confPct}%
          </div>
        </div>
        <div className="bg-bg rounded-lg p-3 border border-border">
          <span className="label">Probability</span>
          <div className="text-2xl font-display font-bold text-accent">{probPct}%</div>
        </div>
        <div className="bg-bg rounded-lg p-3 border border-border">
          <span className="label">Threshold</span>
          <div className="text-2xl font-display font-bold text-text-dim">
            {Math.round(result.threshold * 100)}%
          </div>
        </div>
      </div>

      {/* Prob bar */}
      <div className="mb-5">
        <div className="flex justify-between text-xs font-mono text-muted mb-1.5">
          <span>HAM</span><span>SPAM</span>
        </div>
        <div className="prob-bar">
          <div
            className="prob-bar-fill"
            style={{
              width: `${probPct}%`,
              background: isSpam
                ? 'linear-gradient(90deg, #4ff994, #f9c74f, #f94f4f)'
                : 'linear-gradient(90deg, #4ff994, #4f9cf9)',
            }}
          />
        </div>
        <div className="flex justify-between text-xs font-mono text-muted mt-1">
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-muted text-xs font-mono">
          <Clock size={12} />
          <span>{result.latency_ms}ms</span>
        </div>
        {onExplain && (
          <button
            onClick={onExplain}
            disabled={explaining}
            className="btn-primary flex items-center gap-2 text-xs"
          >
            <Zap size={13} />
            {explaining ? 'Analyzing...' : 'AI Explain'}
          </button>
        )}
      </div>
    </div>
  )
}
