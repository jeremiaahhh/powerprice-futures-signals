import React, { useEffect, useState, useCallback } from 'react'
import { RefreshCw, AlertTriangle, Clock } from 'lucide-react'
import { format, parseISO } from 'date-fns'
import clsx from 'clsx'
import SignalCard from '../components/SignalCard'
import useSignal from '../hooks/useSignal'
import { Signal, SignalAction } from '../api/client'

const ACTION_BADGE: Record<SignalAction, { label: string; cls: string }> = {
  ENTER: { label: 'ENTER', cls: 'bg-[#00ff66]/10 text-[#00ff66] border-[#00ff66]/40' },
  WATCH: { label: 'WATCH', cls: 'bg-[#ffcc00]/10 text-[#ffcc00] border-[#ffcc00]/40' },
  EXIT: { label: 'EXIT', cls: 'bg-[#ff3366]/10 text-[#ff3366] border-[#ff3366]/40' },
  NO_TRADE: { label: 'NO TRADE', cls: 'bg-[#555555]/10 text-[#555555] border-[#555555]/30' },
  RISK: { label: 'RISK', cls: 'bg-[#ff3366]/10 text-[#ff3366] border-[#ff3366]/60' }
}

function CostRow({ label, value, isTotal = false, isNeg = false }: {
  label: string; value: number; isTotal?: boolean; isNeg?: boolean
}) {
  return (
    <tr className={clsx('border-b border-[#1f1f1f]', isTotal && 'bg-[#0a0a0a]')}>
      <td className={clsx('px-3 py-2 text-[11px] font-mono', isTotal ? 'text-[#e8e8e8] font-semibold' : 'text-[#9a9a9a]')}>
        {label}
      </td>
      <td className={clsx('px-3 py-2 text-[11px] font-mono tabular-nums text-right', isTotal ? (isNeg ? 'text-[#ff3366]' : 'text-[#00ff66]') : 'text-[#e8e8e8]', isTotal && 'font-semibold')}>
        {value > 0 ? '+' : ''}€{value.toFixed(2)}
      </td>
    </tr>
  )
}

function FeatureRow({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.max(0, Math.abs(value) * 100))
  const positive = value >= 0

  return (
    <div className="flex items-center gap-3 py-1.5 border-b border-[#1f1f1f]/50 last:border-0">
      <span className="text-[10px] text-[#9a9a9a] font-mono w-40 shrink-0 truncate">{label}</span>
      <div className="flex-1 h-1.5 bg-[#1f1f1f] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            backgroundColor: positive ? '#00ff66' : '#ff3366'
          }}
        />
      </div>
      <span className={clsx('text-[10px] font-mono tabular-nums w-16 text-right', positive ? 'text-[#00ff66]' : 'text-[#ff3366]')}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </span>
    </div>
  )
}

export default function FuturesSignal() {
  const { signal, history, loading, error, lastRefresh, refresh } = useSignal({
    autoRefresh: true,
    intervalMs: 30000,
    includeHistory: true,
    historyLimit: 20
  })

  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true)

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-[#e8e8e8] font-mono">Futures Signal</h1>
          <p className="text-xs text-[#555555] font-mono mt-0.5">
            Negative-price rebound strategy · German power market
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span className="text-[10px] text-[#555555] font-mono flex items-center gap-1">
              <Clock size={10} />
              {format(lastRefresh, 'HH:mm:ss')}
            </span>
          )}
          <label className="flex items-center gap-1.5 text-[10px] text-[#9a9a9a] font-mono cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefreshEnabled}
              onChange={(e) => setAutoRefreshEnabled(e.target.checked)}
              className="w-3 h-3 accent-[#00ff66]"
            />
            Auto-refresh (30s)
          </label>
          <button
            onClick={refresh}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#1f1f1f] rounded text-[10px] text-[#9a9a9a] hover:text-[#e8e8e8] hover:border-[#00ff66]/30 transition-all disabled:opacity-50"
          >
            <RefreshCw size={11} className={clsx(loading && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-3 py-2 bg-[#ffcc00]/10 border border-[#ffcc00]/30 rounded text-[11px] text-[#ffcc00] font-mono">
          <AlertTriangle size={12} />
          API unavailable — showing demo data. {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* Main signal card */}
        <div>
          <h2 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider mb-3">
            Current Signal
          </h2>
          {loading && !signal ? (
            <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg h-64 flex items-center justify-center">
              <div className="text-[#555555] text-xs font-mono animate-pulse">Loading signal…</div>
            </div>
          ) : signal ? (
            <SignalCard signal={signal} />
          ) : (
            <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg p-8 text-center">
              <div className="text-[#555555] text-xs font-mono">No signal data available</div>
            </div>
          )}
        </div>

        {/* Right column: costs + features */}
        <div className="space-y-4">
          {/* Cost breakdown */}
          {signal?.cost_breakdown && (
            <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-[#1f1f1f] bg-[#0a0a0a]">
                <h3 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider">
                  Cost Breakdown
                </h3>
              </div>
              <table className="w-full">
                <tbody>
                  <CostRow label="Bid-Ask Spread" value={-signal.cost_breakdown.spread} />
                  <CostRow label="Slippage" value={-signal.cost_breakdown.slippage} />
                  <CostRow label="Financing / Overnight" value={-signal.cost_breakdown.financing} />
                  <CostRow label="Broker Markup" value={-signal.cost_breakdown.broker_markup} />
                  <CostRow label="Safety Buffer" value={-signal.cost_breakdown.safety_buffer} />
                  <CostRow label="Total Cost" value={-signal.cost_breakdown.total} isTotal isNeg />
                  <CostRow
                    label="Net Edge (after all costs)"
                    value={signal.cost_breakdown.net_edge}
                    isTotal
                    isNeg={signal.cost_breakdown.net_edge < 0}
                  />
                </tbody>
              </table>
            </div>
          )}

          {/* Feature importances */}
          {signal?.features && Object.keys(signal.features).length > 0 && (
            <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-[#1f1f1f] bg-[#0a0a0a]">
                <h3 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider">
                  Feature Contributions
                </h3>
              </div>
              <div className="px-4 py-3 max-h-56 overflow-y-auto scrollbar-thin">
                {Object.entries(signal.features)
                  .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                  .slice(0, 15)
                  .map(([key, val]) => (
                    <FeatureRow
                      key={key}
                      label={key.replace(/_/g, ' ')}
                      value={val}
                    />
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Signal history table */}
      <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[#1f1f1f] bg-[#0a0a0a] flex items-center justify-between">
          <h3 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider">Signal History</h3>
          <span className="text-[10px] text-[#555555] font-mono">Last 20 signals</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[#1f1f1f]">
                {['Timestamp', 'Action', 'Confidence', 'Price', 'Predicted', 'P(Neg)', 'P(Reb)', 'Net Edge'].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-[9px] text-[#555555] uppercase tracking-wider font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-center text-[#555555]">
                    {loading ? 'Loading history…' : 'No signal history available'}
                  </td>
                </tr>
              ) : (
                history.map((s, i) => {
                  const badge = ACTION_BADGE[s.action]
                  return (
                    <tr key={s.id ?? i} className="border-b border-[#1f1f1f]/50 hover:bg-[#0a0a0a] transition-colors">
                      <td className="px-3 py-2 text-[#9a9a9a] whitespace-nowrap">
                        {format(parseISO(s.timestamp), 'MM-dd HH:mm')}
                      </td>
                      <td className="px-3 py-2">
                        <span className={clsx('px-1.5 py-0.5 rounded border text-[9px] font-semibold', badge.cls)}>
                          {badge.label}
                        </span>
                      </td>
                      <td className="px-3 py-2 tabular-nums text-[#e8e8e8]">
                        {(s.confidence * 100).toFixed(0)}%
                      </td>
                      <td className={clsx('px-3 py-2 tabular-nums', s.current_price < 0 ? 'text-[#ff3366]' : 'text-[#e8e8e8]')}>
                        €{s.current_price.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 tabular-nums text-[#00d4ff]">
                        €{s.predicted_price.toFixed(2)}
                      </td>
                      <td className={clsx('px-3 py-2 tabular-nums', s.p_negative > 0.5 ? 'text-[#ff3366]' : 'text-[#9a9a9a]')}>
                        {(s.p_negative * 100).toFixed(0)}%
                      </td>
                      <td className={clsx('px-3 py-2 tabular-nums', s.p_rebound > 0.5 ? 'text-[#00ff66]' : 'text-[#9a9a9a]')}>
                        {(s.p_rebound * 100).toFixed(0)}%
                      </td>
                      <td className={clsx('px-3 py-2 tabular-nums font-semibold', s.net_edge > 0 ? 'text-[#00ff66]' : 'text-[#ff3366]')}>
                        {s.net_edge > 0 ? '+' : ''}€{s.net_edge.toFixed(2)}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
