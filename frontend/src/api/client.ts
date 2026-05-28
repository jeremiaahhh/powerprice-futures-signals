import axios, { AxiosInstance } from 'axios'

// ─── Types ────────────────────────────────────────────────────────────────────

export type SignalAction = 'ENTER' | 'WATCH' | 'EXIT' | 'NO_TRADE' | 'RISK'

export interface CostModel {
  avg_spread: number
  slippage: number
  overnight_fee: number
  broker_markup: number
  safety_buffer: number
  expected_rebound: number
  total_cost: number
  net_edge: number
  is_tradeable: boolean
  rejection_reason?: string
}

export interface ConfidenceBreakdown {
  ml_model: number
  price_level: number
  momentum: number
  volume_signal: number
  weather_factor: number
  overall: number
}

export interface CostBreakdown {
  spread: number
  slippage: number
  financing: number
  broker_markup: number
  safety_buffer: number
  total: number
  net_edge: number
}

export interface Signal {
  id?: string
  timestamp: string
  action: SignalAction
  confidence: number
  confidence_breakdown?: ConfidenceBreakdown
  current_price: number
  predicted_price: number
  p_negative: number
  p_rebound: number
  net_edge: number
  stop_loss: number
  take_profit: number
  reason: string
  risk_warnings?: string[]
  cost_breakdown?: CostBreakdown
  features?: Record<string, number>
  horizon_hours?: number
}

export interface ForecastPoint {
  timestamp: string
  price: number
  p_negative: number
  lower_bound?: number
  upper_bound?: number
}

export interface Forecast {
  generated_at: string
  horizon_hours: number
  points: ForecastPoint[]
  p_rebound_overall: number
  expected_low: number
  expected_high: number
}

export interface BacktestTrade {
  timestamp: string
  action: SignalAction
  entry_price: number
  exit_price: number
  pnl: number
  duration_hours: number
}

export interface BacktestMetrics {
  total_trades: number
  win_rate: number
  profit_factor: number
  total_pnl: number
  max_drawdown: number
  sharpe_ratio: number
  avg_trade_pnl: number
  equity_curve: Array<{ timestamp: string; equity: number }>
  monthly_returns: Array<{ month: string; return_pct: number }>
  trades: BacktestTrade[]
}

export interface BacktestResult {
  naive: BacktestMetrics
  ml: BacktestMetrics
  start_date: string
  end_date: string
}

export interface PaperPosition {
  id: string
  opened_at: string
  action: SignalAction
  entry_price: number
  current_price: number
  unrealized_pnl: number
  size: number
  stop_loss: number
  take_profit: number
}

export interface PaperTrade {
  id: string
  opened_at: string
  closed_at: string
  action: SignalAction
  entry_price: number
  exit_price: number
  realized_pnl: number
  size: number
  signal_confidence: number
  exit_reason: string
}

export interface PaperTradingStatus {
  is_running: boolean
  started_at?: string
  total_pnl: number
  win_rate: number
  total_trades: number
  avg_trade_pnl: number
  open_positions: PaperPosition[]
  trade_journal: PaperTrade[]
  signal_quality: {
    total_signals: number
    enter_signals: number
    watch_signals: number
    no_trade_signals: number
    avg_confidence: number
  }
}

export interface DataSource {
  name: string
  last_update: string
  age_minutes: number
  is_fresh: boolean
  is_warning: boolean
  missing_fields: string[]
  record_count?: number
}

export interface DataQuality {
  sources: {
    smard: DataSource
    entsoe: DataSource
    openmeteo: DataSource
  }
  overall_health: 'healthy' | 'warning' | 'critical'
  last_ingest: string
  age_timeline: Array<{ source: string; age_minutes: number; timestamp: string }>
}

export interface PriceHistory {
  timestamps: string[]
  prices: number[]
  volumes?: number[]
}

export interface CostConfig {
  avg_spread: number
  slippage: number
  overnight_fee: number
  broker_markup: number
  safety_buffer: number
  expected_rebound: number
}

// ─── Axios Instance ───────────────────────────────────────────────────────────

const http: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' }
})

http.interceptors.response.use(
  (res) => res,
  (err) => {
    const message = err.response?.data?.detail ?? err.message ?? 'Unknown error'
    console.error(`[API Error] ${err.config?.url}: ${message}`)
    return Promise.reject(new Error(message))
  }
)

// ─── Response normalisers ─────────────────────────────────────────────────────

/** Map backend SignalAction enum → frontend SignalAction */
function normaliseAction(raw: string): SignalAction {
  if (raw === 'ENTER_LONG_REBOUND_SIGNAL' || raw === 'HIGH_CONFIDENCE_SIGNAL') return 'ENTER'
  if (raw === 'WATCH_LONG_REBOUND') return 'WATCH'
  if (raw === 'EXIT_TAKE_PROFIT_SIGNAL' || raw === 'EXIT_STOP_LOSS_SIGNAL') return 'EXIT'
  if (raw === 'RISK_BLOCKED' || raw === 'DATA_QUALITY_BLOCKED' || raw === 'EXTREME_VOLATILITY_BLOCKED') return 'RISK'
  return 'NO_TRADE'
}

function normaliseSignal(raw: any): Signal {
  return {
    id: String(raw.id ?? raw.timestamp ?? Date.now()),
    timestamp: raw.timestamp ?? raw.generated_at ?? new Date().toISOString(),
    action: normaliseAction(raw.action ?? 'NO_TRADE'),
    confidence: raw.confidence ?? 0,
    current_price: raw.current_price ?? 0,
    predicted_price: raw.predicted_price ?? 0,
    p_negative: raw.p_negative ?? 0,
    p_rebound: raw.p_rebound ?? 0,
    net_edge: raw.net_edge ?? 0,
    stop_loss: raw.stop_loss ?? 0,
    take_profit: raw.take_profit ?? 0,
    reason: raw.reason ?? '',
    risk_warnings: raw.risk_warnings ?? [],
    cost_breakdown: raw.cost_breakdown ?? undefined,
    horizon_hours: raw.max_holding_hours ?? raw.horizon_hours ?? 6,
  }
}

function normaliseBacktestResult(r: any): BacktestMetrics {
  return {
    total_trades: r.total_trades ?? 0,
    win_rate: r.win_rate_pct ?? 0,
    profit_factor: r.profit_factor ?? 0,
    total_pnl: (r.avg_trade_eur_mwh ?? 0) * (r.total_trades ?? 0),
    max_drawdown: r.max_drawdown_pct ?? 0,
    sharpe_ratio: r.sharpe_ratio ?? 0,
    avg_trade_pnl: r.avg_trade_eur_mwh ?? 0,
    equity_curve: (r.equity_curve ?? []).map((p: any) => ({
      timestamp: p.timestamp ?? '',
      equity: p.equity ?? 0,
    })),
    monthly_returns: (r.monthly_performance ?? []).map((m: any) => ({
      month: m.month ?? '',
      return_pct: m.return_pct ?? m.net_pnl_eur ?? 0,
    })),
    trades: (r.trades ?? []).map((t: any) => ({
      timestamp: t.entry_timestamp ?? '',
      action: 'ENTER' as SignalAction,
      entry_price: t.entry_price ?? 0,
      exit_price: t.exit_price ?? 0,
      pnl: t.net_pnl ?? t.pnl_net ?? 0,
      duration_hours: t.holding_hours ?? 0,
    })),
  }
}

// ─── API Functions ────────────────────────────────────────────────────────────

/** Get the latest Futures signal */
export async function getSignal(): Promise<Signal> {
  const res = await http.get<any>('/futures/signal')
  return normaliseSignal(res.data)
}

/** Get signal history (last N signals) */
export async function getSignalHistory(limit = 20): Promise<Signal[]> {
  try {
    const res = await http.get<any[]>('/futures/signal/history', { params: { limit } })
    return res.data.map(normaliseSignal)
  } catch {
    return []
  }
}

/** Get price forecast for next N hours */
export async function getForecast(hours = 6): Promise<Forecast> {
  const res = await http.get<Forecast>('/forecast', { params: { hours } })
  return res.data
}

/** Get historical prices (last N hours) */
export async function getPriceHistory(hours = 24): Promise<PriceHistory> {
  const res = await http.get<any[]>('/data/latest', { params: { hours } })
  const rows: any[] = res.data
  return {
    timestamps: rows.map((r) => r.timestamp),
    prices: rows.map((r) => r.price_eur_mwh ?? 0),
  }
}

/** Get current cost model config */
export async function getCostModel(): Promise<CostModel> {
  const res = await http.get<any>('/futures/cost-model')
  const d = res.data
  return {
    avg_spread: d.avg_spread_eur_mwh ?? 5,
    slippage: d.slippage_eur_mwh ?? 3,
    overnight_fee: d.overnight_fee_annual_pct ?? 8,
    broker_markup: d.broker_markup_eur_mwh ?? 1,
    safety_buffer: d.safety_buffer_eur_mwh ?? 5,
    expected_rebound: 0,
    total_cost: (d.avg_spread_eur_mwh ?? 5) + (d.slippage_eur_mwh ?? 3) + (d.broker_markup_eur_mwh ?? 1) + (d.safety_buffer_eur_mwh ?? 5),
    net_edge: 0,
    is_tradeable: false,
  }
}

/** Update cost model config */
export async function updateCostConfig(config: CostConfig): Promise<CostModel> {
  const payload = {
    avg_spread_eur_mwh: config.avg_spread,
    slippage_eur_mwh: config.slippage,
    overnight_fee_annual_pct: config.overnight_fee,
    broker_markup_eur_mwh: config.broker_markup,
    safety_buffer_eur_mwh: config.safety_buffer,
  }
  const res = await http.post<any>('/futures/cost-model', payload)
  return getCostModel()
}

/** Simulate costs with given parameters — computed locally, no backend call needed */
export async function simulateCosts(config: CostConfig): Promise<CostModel> {
  const totalCost = config.avg_spread + config.slippage + config.broker_markup + config.safety_buffer
  const netEdge = (config.expected_rebound ?? 0) - totalCost
  return {
    ...config,
    total_cost: totalCost,
    net_edge: netEdge,
    is_tradeable: netEdge >= 10,
  }
}

/** Run backtest for date range — returns both naive and ML results */
export async function runBacktest(startDate: string, endDate: string, _strategy?: string): Promise<BacktestResult> {
  const res = await http.get<any>('/backtest/compare-naive-vs-ml', {
    params: { start_date: startDate, end_date: endDate }
  })
  const results: any[] = res.data.results ?? []
  const naive = results.find((r: any) => r.strategy === 'naive') ?? results[0] ?? {}
  const ml = results.find((r: any) => r.strategy === 'ml_rebound') ?? results[1] ?? {}
  return {
    naive: normaliseBacktestResult(naive),
    ml: normaliseBacktestResult(ml),
    start_date: startDate,
    end_date: endDate,
  }
}

/** Get paper trading status */
export async function getPaperTradingStatus(): Promise<PaperTradingStatus> {
  const res = await http.get<any>('/paper/status')
  const d = res.data
  return {
    is_running: d.is_running ?? false,
    started_at: d.started_at,
    total_pnl: d.total_net_pnl_eur ?? 0,
    win_rate: d.win_rate_pct ?? 0,
    total_trades: d.total_closed_positions ?? 0,
    avg_trade_pnl: d.total_closed_positions > 0
      ? (d.total_net_pnl_eur ?? 0) / d.total_closed_positions
      : 0,
    open_positions: (d.positions ?? []).map((p: any) => ({
      id: String(p.id),
      opened_at: p.entry_timestamp ?? p.created_at ?? '',
      action: 'ENTER' as SignalAction,
      entry_price: p.entry_price ?? 0,
      current_price: p.entry_price ?? 0,
      unrealized_pnl: p.pnl_eur ?? 0,
      size: p.notional_size_mwh ?? 1,
      stop_loss: p.stop_loss ?? 0,
      take_profit: p.take_profit ?? 0,
    })),
    trade_journal: [],
    signal_quality: {
      total_signals: 0,
      enter_signals: 0,
      watch_signals: 0,
      no_trade_signals: 0,
      avg_confidence: 0,
    },
  }
}

/** Start paper trading */
export async function startPaperTrading(): Promise<{ message: string }> {
  const res = await http.post<{ message: string }>('/paper/start')
  return res.data
}

/** Stop paper trading */
export async function stopPaperTrading(): Promise<{ message: string }> {
  const res = await http.post<{ message: string }>('/paper/stop')
  return res.data
}

/** Get data quality status */
export async function getDataQuality(): Promise<DataQuality> {
  const res = await http.get<any>('/data/quality')
  const d = res.data
  const src: DataSource = {
    name: 'smard',
    last_update: d.latest_timestamp ?? d.checked_at ?? '',
    age_minutes: d.age_minutes ?? 0,
    is_fresh: d.is_fresh ?? false,
    is_warning: !d.is_fresh,
    missing_fields: d.missing_fields ?? [],
    record_count: d.row_count_last_24h,
  }
  const health = d.is_fresh ? 'healthy' : d.age_minutes < 240 ? 'warning' : 'critical'
  return {
    sources: { smard: src, entsoe: { ...src, name: 'entsoe' }, openmeteo: { ...src, name: 'openmeteo' } },
    overall_health: health as DataQuality['overall_health'],
    last_ingest: d.checked_at ?? new Date().toISOString(),
    age_timeline: [],
  }
}

/** Trigger manual data ingest */
export async function triggerIngest(source?: string): Promise<{ message: string; status: string }> {
  const res = await http.post<any>('/data/ingest', { source: source ?? 'smard' })
  return { message: `Upserted ${res.data.rows_inserted ?? 0} rows`, status: 'ok' }
}

/** Health check */
export async function healthCheck(): Promise<{ status: string; timestamp: string }> {
  const res = await http.get<{ status: string; timestamp: string }>('/health')
  return res.data
}

// ─── Battery Types ────────────────────────────────────────────────────────────

export interface BatteryCapacity {
  power_mw: number
  capacity_mwh: number
  source: string
  as_of: string
  data_quality_score: number
}

export interface BatteryLatest {
  capacity: BatteryCapacity
  latest_proxy: {
    timestamp: string
    battery_saturation_proxy: number | null
    storage_charge_pressure: number | null
    storage_discharge_pressure: number | null
    net_battery_flow_mw: number | null
    battery_charging_mw: number | null
    battery_discharging_mw: number | null
    pv_surplus_after_load: number | null
    midday_price_compression: number | null
    evening_arbitrage_spread: number | null
  } | null
  generated_at: string
}

export interface BatteryFlowPoint {
  timestamp: string
  charging_mw: number
  discharging_mw: number
  net_battery_flow_mw: number
  source: string
  is_proxy: boolean
  data_quality_score: number
}

export interface BatteryDataQuality {
  status: string
  entsoe_key_configured: boolean
  data_source: string
  is_proxy: boolean
  data_quality_score: number
  installed_capacity: BatteryCapacity
  battery_columns: number
  note: string
}

export interface BatteryRegimeImpact {
  regime: {
    regime: string
    confidence: number
    signal_thresholds: { net_edge_enter: number; net_edge_hc: number }
    description: string
  }
  battery_state: {
    battery_saturation_proxy: number | null
    storage_charge_pressure: number | null
    storage_discharge_pressure: number | null
    net_battery_flow_mw: number | null
    expected_battery_absorption: number | null
    expected_battery_release: number | null
  }
  hc_signal_blocked: boolean
  generated_at: string
}

// ─── Battery API Functions ────────────────────────────────────────────────────

export async function getBatteryLatest(): Promise<BatteryLatest> {
  const res = await http.get<BatteryLatest>('/battery/latest')
  return res.data
}

export async function getBatteryCapacity(): Promise<BatteryCapacity> {
  const res = await http.get<BatteryCapacity>('/battery/capacity')
  return res.data
}

export async function getBatteryFlows(hours = 48): Promise<BatteryFlowPoint[]> {
  try {
    const res = await http.get<BatteryFlowPoint[]>('/battery/flows', { params: { hours } })
    return res.data
  } catch {
    return []
  }
}

export async function getBatteryDataQuality(): Promise<BatteryDataQuality> {
  const res = await http.get<BatteryDataQuality>('/battery/data-quality')
  return res.data
}

export async function getBatteryRegimeImpact(): Promise<BatteryRegimeImpact> {
  const res = await http.get<BatteryRegimeImpact>('/battery/regime-impact')
  return res.data
}

// ─── Risk module interfaces ──────────────────────────────────────────────────
const BASE_URL = '/api'

export interface TailRiskAssessment {
  tail_risk_score: number
  gap_risk_score: number
  oversupply_stress_index: number
  rebound_failure_probability: number
  negative_price_streak: number
  max_price_gap_1h: number
  volatility_24h: number
  is_blocked: boolean
  block_reason: string | null
  block_detail: string
  components: Record<string, number>
  current_price: number
  generated_at: string
}

export interface GapAssessment {
  max_gap_1h: number
  gap_score: number
  has_extreme_gap: boolean
  gap_timestamps: string[]
  threshold_eur_mwh: number
  generated_at: string
}

export interface VolatilityAssessment {
  vol_1h: number | null
  vol_6h: number
  vol_24h: number
  vol_spike_ratio: number
  regime: 'NORMAL' | 'ELEVATED' | 'EXTREME'
  is_blocked: boolean
  detail: string
  generated_at: string
}

export interface OOSPerformanceSummary {
  ml_runs_count: number
  avg_sharpe: number | null
  avg_win_rate: number | null
  avg_profit_factor: number | null
}

export interface OOSPerformanceRun {
  run_id: string
  strategy: string
  start_date: string | null
  end_date: string | null
  total_trades: number | null
  win_rate_pct: number | null
  sharpe_ratio: number | null
  max_drawdown_pct: number | null
  total_return_pct: number | null
  profit_factor: number | null
  worst_trade_eur_mwh: number | null
  created_at: string | null
}

export interface OOSPerformance {
  status: string
  summary: OOSPerformanceSummary
  runs: OOSPerformanceRun[]
  generated_at: string
}

export const getRiskTail = (): Promise<TailRiskAssessment> =>
  fetch(`${BASE_URL}/risk/tail`).then(r => r.json())

export const getRiskGap = (): Promise<GapAssessment> =>
  fetch(`${BASE_URL}/risk/gap`).then(r => r.json())

export const getRiskVolatility = (): Promise<VolatilityAssessment> =>
  fetch(`${BASE_URL}/risk/volatility`).then(r => r.json())

export const getOOSPerformance = (): Promise<OOSPerformance> =>
  fetch(`${BASE_URL}/analytics/oos-performance`).then(r => r.json())

export const getRegimeDrift = (days = 30): Promise<any> =>
  fetch(`${BASE_URL}/analytics/regime-drift?days=${days}`).then(r => r.json())

export const getBlockedTrades = (): Promise<any[]> =>
  fetch(`${BASE_URL}/risk/blocked-trades`).then(r => r.json())

// ============================================================
// Daemon interfaces
// ============================================================

export interface DaemonStatus {
  running: boolean;
  pid: number | null;
  started_at: string | null;
  last_run_at: string | null;
  next_run_at: string | null;
  cycle_count: number;
  consecutive_errors: number;
  last_error: string | null;
  last_signal: string | null;
  last_signal_at: string | null;
  telegram_enabled: boolean;
  auto_retrain_enabled: boolean;
  signal_mode: string;
  stop_signal_pending: boolean;
  telegram_sent_today: number;
  generated_at: string;
}

export interface NotificationEvent {
  id: number;
  created_at: string;
  channel: string;
  event_type: string;
  signal_id: number | null;
  fingerprint: string | null;
  payload: Record<string, unknown> | null;
  status: string;
  error_message: string | null;
}

export interface NotificationStats {
  days_analyzed: number;
  total: number;
  sent: number;
  failed: number;
  by_type: Record<string, number>;
  last_sent_at: string | null;
}

export interface DriftReport {
  has_drift: boolean;
  severity: string;
  drift_types: string[];
  details: Record<string, unknown>;
  checked_at: string;
}

export interface ModelRegistryStatus {
  production_metrics: Record<string, unknown>;
  candidate_count: number;
  last_candidates: Array<Record<string, unknown>>;
  last_promotion: Record<string, unknown>;
}

export interface RollingPerformance {
  rolling_pf: number | null;
  rolling_win_rate: number | null;
  sample_size: number;
  generated_at: string;
}

export interface ShadowOutcome {
  id: number;
  signal_id: number;
  evaluated_at: string;
  realized_price_1h: number | null;
  realized_price_2h: number | null;
  realized_price_4h: number | null;
  realized_rebound: number | null;
  simulated_pnl: number | null;
  would_hit_stop: boolean | null;
  would_hit_take_profit: boolean | null;
  outcome_status: string | null;
}

// ============================================================
// Daemon API functions
// ============================================================

export const getDaemonStatus = (): Promise<DaemonStatus> =>
  fetch(`${BASE_URL}/daemon/status`).then(r => r.json());

export const postDaemonStop = (): Promise<Record<string, string>> =>
  fetch(`${BASE_URL}/daemon/stop`, { method: 'POST' }).then(r => r.json());

export const postDaemonStart = (): Promise<Record<string, string>> =>
  fetch(`${BASE_URL}/daemon/start`, { method: 'POST' }).then(r => r.json());

export const postDaemonRestart = (): Promise<Record<string, string>> =>
  fetch(`${BASE_URL}/daemon/restart`, { method: 'POST' }).then(r => r.json());

export const getDaemonLogs = (lines?: number): Promise<{ lines: string[]; total_lines: number }> =>
  fetch(`${BASE_URL}/daemon/logs${lines ? `?lines=${lines}` : ''}`).then(r => r.json());

export const getDaemonLastRun = (): Promise<Record<string, unknown>> =>
  fetch(`${BASE_URL}/daemon/last-run`).then(r => r.json());

// ============================================================
// Notification API functions
// ============================================================

export const getRecentNotifications = (limit?: number): Promise<NotificationEvent[]> =>
  fetch(`${BASE_URL}/notifications/recent${limit ? `?limit=${limit}` : ''}`).then(r => r.json());

export const getNotificationStats = (days?: number): Promise<NotificationStats> =>
  fetch(`${BASE_URL}/notifications/stats${days ? `?days=${days}` : ''}`).then(r => r.json());

// ============================================================
// Adaptation API functions
// ============================================================

export const getDriftReport = (): Promise<DriftReport> =>
  fetch(`${BASE_URL}/adaptation/drift-report`).then(r => r.json());

export const getModelRegistry = (): Promise<ModelRegistryStatus> =>
  fetch(`${BASE_URL}/adaptation/model-registry`).then(r => r.json());

export const getThresholdAnalysis = (days?: number): Promise<Record<string, unknown>> =>
  fetch(`${BASE_URL}/adaptation/threshold-analysis${days ? `?days=${days}` : ''}`).then(r => r.json());

export const getRollingPerformance = (window?: number): Promise<RollingPerformance> =>
  fetch(`${BASE_URL}/adaptation/rolling-performance${window ? `?window=${window}` : ''}`).then(r => r.json());

// ============================================================
// Shadow outcomes API functions
// ============================================================

export const getShadowOutcomes = (limit?: number): Promise<ShadowOutcome[]> =>
  fetch(`${BASE_URL}/shadow/outcomes${limit ? `?limit=${limit}` : ''}`).then(r => r.json());
