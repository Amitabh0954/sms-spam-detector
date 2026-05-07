import { NavLink, useLocation } from 'react-router-dom'
import { Shield, MessageSquare, LayoutList, BarChart3, Brain, Activity } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const navItems = [
  { to: '/',        icon: MessageSquare, label: 'Predict',    sub: 'Single message' },
  { to: '/batch',   icon: LayoutList,    label: 'Batch',      sub: 'Multi-message' },
  { to: '/models',  icon: BarChart3,     label: 'Models',     sub: 'Compare & metrics' },
  { to: '/explain', icon: Brain,         label: 'Intelligence', sub: 'AI explain' },
]

export default function Layout({ children }) {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'error' }))
  }, [])

  return (
    <div className="noise-bg flex min-h-screen">
      <div className="scanline" />

      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-border flex flex-col bg-surface/60 backdrop-blur-sm fixed h-full z-10">
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-accent/10 border border-accent/30 flex items-center justify-center glow-accent">
              <Shield size={18} className="text-accent" />
            </div>
            <div>
              <div className="font-display font-700 text-text text-base leading-tight">SpamShield</div>
              <div className="text-text-dim text-xs font-mono">v3.0 · Groq LLM</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label, sub }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group ${
                  isActive
                    ? 'bg-accent/10 border border-accent/20 text-accent'
                    : 'text-text-dim hover:text-text hover:bg-border/50'
                }`
              }
            >
              <Icon size={16} />
              <div>
                <div className="text-sm font-medium leading-tight">{label}</div>
                <div className="text-xs text-muted group-hover:text-text-dim transition-colors">{sub}</div>
              </div>
            </NavLink>
          ))}
        </nav>

        {/* Status */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-success animate-pulse' : 'bg-danger'}`} />
            <span className="text-xs font-mono text-text-dim">
              {health?.status === 'ok' ? 'API Connected' : health?.status === 'error' ? 'API Offline' : 'Connecting...'}
            </span>
          </div>
          {health?.models && (
            <div className="mt-2 text-xs text-muted font-mono">
              {health.models.length} models loaded
            </div>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="ml-64 flex-1 min-h-screen">
        <div className="max-w-5xl mx-auto p-8 animate-fade-in">
          {children}
        </div>
      </main>
    </div>
  )
}
