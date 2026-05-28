import React, { useState, useEffect } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import clsx from 'clsx'
import { healthCheck } from '../api/client'
import { format } from 'date-fns'

interface NavItem {
  code: string
  label: string
  path: string
}

const NAV_ITEMS: NavItem[] = [
  { code: 'OVR',  label: 'Overview',       path: '/' },
  { code: 'SGNL', label: 'Futures Signal',     path: '/signal' },
  { code: 'FCST', label: 'Forecast',       path: '/forecast' },
  { code: 'COST', label: 'Cost Simulator', path: '/costs' },
  { code: 'BCKT', label: 'Backtest',       path: '/backtest' },
  { code: 'PAPR', label: 'Paper Trading',  path: '/paper' },
  { code: 'DATA', label: 'Data Quality',   path: '/data' },
  { code: 'BATT', label: 'Battery Intel.', path: '/battery' },
  { code: 'TAIL', label: 'Tail Risk',      path: '/tail-risk' },
  { code: 'STBL', label: 'OOS Stability',  path: '/signal-stability' },
  { code: 'DEMN', label: 'Daemon',         path: '/daemon' },
  { code: 'TGRM', label: 'Telegram',       path: '/telegram' },
  { code: 'SHDW', label: 'Shadow Mode',    path: '/shadow-mode' },
  { code: 'RTRN', label: 'Retraining',     path: '/auto-retraining' },
  { code: 'DRFT', label: 'Drift Monitor',  path: '/drift-monitor' },
]

export default function Layout() {
  const [now, setNow] = useState(new Date())
  const [connected, setConnected] = useState<boolean | null>(null)

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        await healthCheck()
        if (!cancelled) setConnected(true)
      } catch {
        if (!cancelled) setConnected(false)
      }
    }
    check()
    const id = setInterval(check, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const statusText =
    connected === null ? 'CONNECTING' : connected ? 'LIVE' : 'OFFLINE'
  const statusColor =
    connected === null ? 'text-warn'
      : connected      ? 'text-bull'
      : 'text-bear'

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-black text-text-primary">
      {/* ============================================================== */}
      {/* TOP STATUS BAR — black, amber accent, fixed height            */}
      {/* ============================================================== */}
      <header className="flex-shrink-0 h-7 flex items-center px-3 border-b border-term-border bg-term-bg text-2xs">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-amber font-semibold tracking-wider">PWRP</span>
          <span className="text-text-muted">|</span>
          <span className="text-text-secondary uppercase tracking-wider">
            German Power · EPEX Day-Ahead
          </span>
          <span className="text-text-muted">|</span>
          <span className="text-text-secondary uppercase">Signal Only</span>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-text-muted uppercase tracking-wider">
            {format(now, 'yyyy-MM-dd')}
          </span>
          <span className="text-amber font-semibold tabular-nums">
            {format(now, 'HH:mm:ss')} CET
          </span>
          <span className={clsx('font-semibold tracking-wider', statusColor)}>
            <span className={clsx('inline-block w-1.5 h-1.5 mr-1.5 align-middle',
              connected === null && 'bg-warn term-blink',
              connected       && 'bg-bull',
              connected === false && 'bg-bear'
            )} />
            {statusText}
          </span>
        </div>
      </header>

      {/* ============================================================== */}
      {/* DISCLAIMER STRIP                                              */}
      {/* ============================================================== */}
      <div className="flex-shrink-0 h-6 flex items-center px-3 bg-amber/10 border-b border-amber/30 text-3xs">
        <span className="text-amber font-semibold tracking-wider uppercase">
          NOTICE — Signal Only. Not financial advice. Futures trading involves substantial risk of loss.
        </span>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* ============================================================ */}
        {/* SIDEBAR — code+label nav, no icons, no rounding             */}
        {/* ============================================================ */}
        <aside className="flex-shrink-0 w-44 bg-term-bg border-r border-term-border overflow-y-auto scrollbar-thin">
          <div className="px-3 py-2 border-b border-term-border">
            <div className="text-3xs text-text-muted uppercase tracking-wider">Modules</div>
          </div>
          <nav>
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-2 px-3 py-1.5 text-2xs border-l-2 transition-colors',
                    isActive
                      ? 'bg-term-panel border-l-amber text-text-primary'
                      : 'border-l-transparent text-text-secondary hover:bg-term-panel hover:text-text-primary'
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={clsx(
                        'w-10 text-3xs uppercase tracking-wider tabular-nums',
                        isActive ? 'text-amber' : 'text-text-muted'
                      )}
                    >
                      {item.code}
                    </span>
                    <span className="truncate">{item.label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="px-3 py-2 mt-2 border-t border-term-border text-3xs text-text-muted uppercase tracking-wider">
            <div className="flex justify-between">
              <span>Build</span>
              <span className="text-text-secondary">v1.0</span>
            </div>
            <div className="flex justify-between mt-0.5">
              <span>Market</span>
              <span className="text-text-secondary">DE-LU</span>
            </div>
          </div>
        </aside>

        {/* ============================================================ */}
        {/* MAIN                                                         */}
        {/* ============================================================ */}
        <main className="flex-1 overflow-y-auto scrollbar-thin bg-black">
          <div className="p-4">
            <Outlet />
          </div>
        </main>
      </div>

      {/* ============================================================== */}
      {/* BOTTOM COMMAND STRIP                                          */}
      {/* ============================================================== */}
      <footer className="flex-shrink-0 h-5 flex items-center px-3 border-t border-term-border bg-term-bg text-3xs text-text-muted uppercase tracking-wider">
        <span>SMARD</span>
        <span className="mx-2 text-text-dim">|</span>
        <span>ENTSO-E</span>
        <span className="mx-2 text-text-dim">|</span>
        <span>Open-Meteo</span>
        <span className="ml-auto">Read-only research interface</span>
      </footer>
    </div>
  )
}
