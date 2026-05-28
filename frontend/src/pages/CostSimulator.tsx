import React, { useState, useCallback, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine
} from 'recharts'
import { Save, Calculator, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react'
import clsx from 'clsx'
import { getCostModel, simulateCosts, updateCostConfig, CostModel, CostConfig } from '../api/client'

interface SliderInputProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit: string
  onChange: (v: number) => void
  description?: string
  color?: string
}

function SliderInput({ label, value, min, max, step, unit, onChange, description, color = '#00d4ff' }: SliderInputProps) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-[#9a9a9a] uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-1">
          <input
            type="number"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
            className="w-20 bg-[#000000] border border-[#1f1f1f] rounded px-2 py-0.5 text-[11px] font-mono text-[#e8e8e8] text-right focus:outline-none focus:border-[#00d4ff] transition-colors"
          />
          <span className="text-[10px] text-[#555555] font-mono">{unit}</span>
        </div>
      </div>
      <div className="relative">
        <div className="h-1.5 bg-[#1f1f1f] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-150"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
      </div>
      {description && (
        <p className="text-[9px] text-[#555555] font-mono">{description}</p>
      )}
    </div>
  )
}

const DEFAULT_CONFIG: CostConfig = {
  avg_spread: 1.5,
  slippage: 0.8,
  overnight_fee: 0.15,
  broker_markup: 0.5,
  safety_buffer: 2.0,
  expected_rebound: 18.0
}

export default function CostSimulator() {
  const [config, setConfig] = useState<CostConfig>(DEFAULT_CONFIG)
  const [result, setResult] = useState<CostModel | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const simulate = useCallback(async (cfg: CostConfig) => {
    setLoading(true)
    try {
      const res = await simulateCosts(cfg)
      setResult(res)
      setError(null)
    } catch {
      // Calculate locally as fallback
      const total = cfg.avg_spread + cfg.slippage + cfg.overnight_fee + cfg.broker_markup + cfg.safety_buffer
      const net = cfg.expected_rebound - total
      setResult({
        ...cfg,
        total_cost: total,
        net_edge: net,
        is_tradeable: net > 0,
        rejection_reason: net <= 0 ? 'Net edge is negative — costs exceed expected rebound' : undefined
      })
    } finally {
      setLoading(false)
    }
  }, [])

  // Load current config on mount
  useEffect(() => {
    const load = async () => {
      try {
        const model = await getCostModel()
        setConfig({
          avg_spread: model.avg_spread,
          slippage: model.slippage,
          overnight_fee: model.overnight_fee,
          broker_markup: model.broker_markup,
          safety_buffer: model.safety_buffer,
          expected_rebound: model.expected_rebound
        })
        setResult(model)
      } catch {
        simulate(DEFAULT_CONFIG)
      }
    }
    load()
  }, [simulate])

  // Debounce simulation on config change
  useEffect(() => {
    const timer = setTimeout(() => simulate(config), 300)
    return () => clearTimeout(timer)
  }, [config, simulate])

  const handleSave = async () => {
    setSaving(true)
    try {
      const res = await updateCostConfig(config)
      setResult(res)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
      setError(null)
    } catch (err) {
      setError('Failed to save config — API unavailable')
    } finally {
      setSaving(false)
    }
  }

  const updateConfig = (key: keyof CostConfig) => (v: number) => {
    setConfig(prev => ({ ...prev, [key]: v }))
  }

  // Chart data
  const costItems = result ? [
    { name: 'Spread', value: config.avg_spread, type: 'cost' },
    { name: 'Slippage', value: config.slippage, type: 'cost' },
    { name: 'Financing', value: config.overnight_fee, type: 'cost' },
    { name: 'Broker', value: config.broker_markup, type: 'cost' },
    { name: 'Buffer', value: config.safety_buffer, type: 'cost' },
    { name: 'Rebound', value: config.expected_rebound, type: 'rebound' }
  ] : []

  const isPositive = result && result.net_edge > 0

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-[#e8e8e8] font-mono">Cost Simulator</h1>
          <p className="text-xs text-[#555555] font-mono mt-0.5">
            Adjust cost parameters and see live net edge calculation
          </p>
        </div>
        <div className="flex items-center gap-2">
          {saveSuccess && (
            <span className="flex items-center gap-1 text-[10px] text-[#00ff66] font-mono">
              <CheckCircle size={11} /> Saved
            </span>
          )}
          {error && (
            <span className="text-[10px] text-[#ffcc00] font-mono">{error}</span>
          )}
          <button
            onClick={() => setConfig(DEFAULT_CONFIG)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#1f1f1f] rounded text-[10px] text-[#9a9a9a] hover:text-[#e8e8e8] transition-all"
          >
            <RefreshCw size={11} />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#00d4ff]/10 border border-[#00d4ff]/40 rounded text-[10px] text-[#00d4ff] hover:bg-[#00d4ff]/20 transition-all disabled:opacity-50"
          >
            <Save size={11} className={clsx(saving && 'animate-spin')} />
            {saving ? 'Saving…' : 'Save Config'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* Sliders */}
        <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg p-5 space-y-5">
          <div className="flex items-center gap-2 mb-2">
            <Calculator size={14} className="text-[#9a9a9a]" />
            <h2 className="text-xs font-semibold text-[#e8e8e8] font-mono uppercase tracking-wider">
              Cost Parameters
            </h2>
          </div>

          <div className="border-t border-[#1f1f1f] pt-4 space-y-4">
            <div className="text-[9px] text-[#555555] font-mono uppercase tracking-wider">Trading Costs</div>
            <SliderInput
              label="Avg Spread"
              value={config.avg_spread}
              min={0} max={10} step={0.1}
              unit="€/MWh"
              onChange={updateConfig('avg_spread')}
              description="Typical bid-ask spread for DE power Futures"
              color="#ff3366"
            />
            <SliderInput
              label="Slippage"
              value={config.slippage}
              min={0} max={5} step={0.05}
              unit="€/MWh"
              onChange={updateConfig('slippage')}
              description="Expected execution slippage at market open"
              color="#ff3366"
            />
            <SliderInput
              label="Overnight Fee"
              value={config.overnight_fee}
              min={0} max={2} step={0.01}
              unit="€/MWh"
              onChange={updateConfig('overnight_fee')}
              description="Daily financing / swap cost"
              color="#ffcc00"
            />
            <SliderInput
              label="Broker Markup"
              value={config.broker_markup}
              min={0} max={3} step={0.05}
              unit="€/MWh"
              onChange={updateConfig('broker_markup')}
              description="Broker commission and markup"
              color="#ffcc00"
            />
            <SliderInput
              label="Safety Buffer"
              value={config.safety_buffer}
              min={0} max={10} step={0.25}
              unit="€/MWh"
              onChange={updateConfig('safety_buffer')}
              description="Extra margin for unforeseen costs"
              color="#ffa500"
            />
          </div>

          <div className="border-t border-[#1f1f1f] pt-4 space-y-4">
            <div className="text-[9px] text-[#555555] font-mono uppercase tracking-wider">Expected Return</div>
            <SliderInput
              label="Expected Rebound"
              value={config.expected_rebound}
              min={0} max={80} step={0.5}
              unit="€/MWh"
              onChange={updateConfig('expected_rebound')}
              description="Estimated price recovery from negative to positive"
              color="#00ff66"
            />
          </div>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Net edge result */}
          <div
            className={clsx(
              'rounded-lg border-2 p-5 transition-all',
              isPositive ? 'bg-[#00ff66]/5 border-[#00ff66]/40' : 'bg-[#ff3366]/5 border-[#ff3366]/40'
            )}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider">
                Net Edge Result
              </h3>
              <div
                className={clsx(
                  'flex items-center gap-1.5 px-2.5 py-1 rounded border text-[10px] font-mono font-semibold',
                  isPositive
                    ? 'text-[#00ff66] border-[#00ff66]/40 bg-[#00ff66]/10'
                    : 'text-[#ff3366] border-[#ff3366]/40 bg-[#ff3366]/10'
                )}
              >
                {isPositive ? <CheckCircle size={11} /> : <AlertTriangle size={11} />}
                {isPositive ? 'TRADEABLE' : 'NOT TRADEABLE'}
              </div>
            </div>

            {result && (
              <>
                <div className="text-center mb-4">
                  <div className="text-[10px] text-[#555555] font-mono mb-1">Net Edge After All Costs</div>
                  <div className={clsx('text-4xl font-mono font-bold tabular-nums', isPositive ? 'text-[#00ff66]' : 'text-[#ff3366]')}>
                    {result.net_edge > 0 ? '+' : ''}€{result.net_edge.toFixed(2)}
                  </div>
                  <div className="text-[10px] text-[#555555] font-mono">per MWh</div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
                  <div className="bg-[#000000]/50 rounded p-2">
                    <div className="text-[9px] text-[#555555] uppercase mb-1">Expected Rebound</div>
                    <div className="text-[#00ff66] font-semibold">+€{config.expected_rebound.toFixed(2)}</div>
                  </div>
                  <div className="bg-[#000000]/50 rounded p-2">
                    <div className="text-[9px] text-[#555555] uppercase mb-1">Total Cost</div>
                    <div className="text-[#ff3366] font-semibold">-€{result.total_cost.toFixed(2)}</div>
                  </div>
                </div>

                {result.rejection_reason && (
                  <div className="mt-3 flex items-start gap-2 text-[10px] text-[#ff3366] font-mono bg-[#ff3366]/10 rounded px-3 py-2">
                    <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                    {result.rejection_reason}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Cost breakdown visual */}
          <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg p-4">
            <h3 className="text-[10px] font-mono text-[#555555] uppercase tracking-wider mb-4">
              Cost vs Rebound Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={costItems} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#555555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                  tickLine={false}
                  axisLine={{ stroke: '#1f1f1f' }}
                />
                <YAxis
                  tick={{ fill: '#555555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `€${v}`}
                  width={40}
                />
                <Tooltip
                  formatter={(v: number) => [`€${v.toFixed(2)}`, '']}
                  contentStyle={{ background: '#111111', border: '1px solid #1f1f1f', borderRadius: 6, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  labelStyle={{ color: '#9a9a9a' }}
                />
                <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                  {costItems.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.type === 'rebound' ? '#00ff66' : entry.name === 'Buffer' ? '#ffa500' : entry.name === 'Overnight' || entry.name === 'Broker' ? '#ffcc00' : '#ff3366'}
                      fillOpacity={0.85}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Break-even */}
          <div className="bg-[#111111] border border-[#1f1f1f] rounded-lg px-4 py-3">
            <div className="text-[9px] text-[#555555] font-mono uppercase tracking-wider mb-2">Break-Even Analysis</div>
            <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
              <div>
                <div className="text-[9px] text-[#555555] mb-0.5">Min Rebound to Trade</div>
                <div className="text-[#ffcc00] font-semibold">
                  {result ? `€${result.total_cost.toFixed(2)}` : '—'} / MWh
                </div>
              </div>
              <div>
                <div className="text-[9px] text-[#555555] mb-0.5">Edge / Cost Ratio</div>
                <div className={clsx('font-semibold', isPositive ? 'text-[#00ff66]' : 'text-[#ff3366]')}>
                  {result && result.total_cost > 0
                    ? `${((result.net_edge / result.total_cost) * 100).toFixed(0)}%`
                    : '—'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
