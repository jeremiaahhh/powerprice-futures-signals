import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import FuturesSignal from './pages/FuturesSignal'
import Forecast from './pages/Forecast'
import CostSimulator from './pages/CostSimulator'
import Backtest from './pages/Backtest'
import PaperTrading from './pages/PaperTrading'
import DataQuality from './pages/DataQuality'
import BatteryIntelligence from './pages/BatteryIntelligence'
import TailRiskMonitor from './pages/TailRiskMonitor'
import SignalStability from './pages/SignalStability'
import DaemonStatus from './pages/DaemonStatus'
import TelegramSettings from './pages/TelegramSettings'
import ShadowMode from './pages/ShadowMode'
import AutoRetraining from './pages/AutoRetraining'
import DriftMonitor from './pages/DriftMonitor'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="signal" element={<FuturesSignal />} />
        <Route path="forecast" element={<Forecast />} />
        <Route path="costs" element={<CostSimulator />} />
        <Route path="backtest" element={<Backtest />} />
        <Route path="paper" element={<PaperTrading />} />
        <Route path="data" element={<DataQuality />} />
        <Route path="battery" element={<BatteryIntelligence />} />
        <Route path="tail-risk" element={<TailRiskMonitor />} />
        <Route path="signal-stability" element={<SignalStability />} />
        <Route path="daemon" element={<DaemonStatus />} />
        <Route path="telegram" element={<TelegramSettings />} />
        <Route path="shadow-mode" element={<ShadowMode />} />
        <Route path="auto-retraining" element={<AutoRetraining />} />
        <Route path="drift-monitor" element={<DriftMonitor />} />
        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
