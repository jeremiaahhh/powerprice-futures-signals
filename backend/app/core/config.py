from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        protected_namespaces=("settings_",),
    )

    # App
    app_name: str = "PowerPrice Futures Signals"
    app_env: str = "development"
    signal_only: bool = True
    secret_key: str = "change-me"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://ppuser:pppass@postgres:5432/powerprice"
    database_url_sync: str = "postgresql+psycopg2://ppuser:pppass@postgres:5432/powerprice"

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # Data Sources
    entsoe_api_key: str = ""
    smard_base_url: str = "https://www.smard.de/app/chart_data"
    entsoe_base_url: str = "https://transparency.entsoe.eu/api"
    openmeteo_base_url: str = "https://api.open-meteo.com/v1/forecast"

    # ML
    model_dir: str = "/app/models"
    retrain_interval_hours: int = 24

    # Futures Defaults
    futures_avg_spread_eur_mwh: float = 5.0
    futures_min_spread_eur_mwh: float = 2.0
    futures_max_spread_eur_mwh: float = 15.0
    futures_volatility_spread_multiplier: float = 1.5
    futures_slippage_eur_mwh: float = 3.0
    futures_overnight_fee_annual_pct: float = 8.0
    futures_weekend_fee_multiplier: float = 1.5
    futures_broker_markup_eur_mwh: float = 1.0
    futures_safety_buffer_eur_mwh: float = 5.0
    futures_min_edge_threshold: float = 30.0
    futures_high_confidence_threshold: float = 35.0
    futures_watch_threshold: float = 20.0

    # Risk
    max_risk_per_signal_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 3
    max_spread_filter_eur_mwh: float = 12.0
    min_confidence_threshold: float = 0.60
    max_data_age_minutes: int = 90

    # Tail Risk Engine
    max_negative_price_eur_mwh: float = -150.0
    max_negative_streak_hours: int = 3
    max_gap_size_eur_mwh: float = 100.0
    max_tail_risk_score: float = 0.65
    tail_risk_extreme_vol_threshold: float = 150.0
    tail_risk_vol_spike_multiplier: float = 2.5

    # Raised entry thresholds (from OOS analysis, 2026-05-20)
    futures_p_rebound_entry: float = 0.70
    futures_p_rebound_watch: float = 0.55

    # Daemon
    signal_daemon_enabled: bool = True
    signal_loop_interval_minutes: int = 15
    data_stale_max_minutes: int = 90
    max_consecutive_errors: int = 5
    daemon_state_file: str = "/app/data/daemon_health.json"
    daemon_stop_signal_file: str = "/app/data/daemon_stop.signal"
    daemon_log_file: str = "/app/data/daemon_run.log"

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_min_signal_level: str = "WATCH"  # WATCH, ENTER, HIGH_CONFIDENCE
    telegram_send_blocked_signals: bool = False
    telegram_send_daily_summary: bool = True
    signal_dedup_minutes: int = 60

    # Auto retraining
    auto_retrain_enabled: bool = True
    rolling_training_days: int = 365
    drift_check_enabled: bool = True
    drift_check_interval_hours: int = 6
    time_decay_enabled: bool = False
    time_decay_half_life_days: int = 180

    # Signal mode (NORMAL or WATCH_ONLY)
    signal_mode: str = "NORMAL"
    signal_mode_rolling_window: int = 30
    signal_mode_pf_floor: float = 1.0
    signal_mode_pf_recovery: float = 1.3

    # Shadow mode
    shadow_mode_enabled: bool = True

    # Sleep / energy-saving mode
    signal_loop_interval_battery_minutes: int = 30
    wake_detection_enabled: bool = True
    post_wake_grace_seconds: int = 15
    wake_clock_jump_threshold_seconds: float = 30.0


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
