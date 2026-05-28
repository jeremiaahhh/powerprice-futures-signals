"""
Thread-safe singleton model registry for the PowerPrice Futures Signals platform.

Loads trained model artefacts from disk on first access and caches them
in memory for the lifetime of the process.  All three models
(NegativePriceClassifier, ReboundClassifier, PriceRegressionModel) are
surfaced via typed accessors.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel

logger = get_logger(__name__)

_METRICS_FILE = "training_metrics.json"
_LAST_TRAINED_FILE = "last_trained.json"


class ModelRegistry:
    """Singleton for loading and caching trained ML model artefacts.

    Usage
    -----
    .. code-block:: python

        registry = ModelRegistry.get_instance()
        if registry.are_models_ready():
            p_neg = registry.get_negative_classifier().predict_proba(features)

    All accessors raise ``RuntimeError`` if the requested model has not
    been successfully loaded (e.g. artefacts are missing on disk).  Call
    :meth:`are_models_ready` before production use to avoid surprises.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._model_dir: str = settings.model_dir
        self._neg_clf: Optional[NegativePriceClassifier] = None
        self._reb_clf: Optional[ReboundClassifier] = None
        self._reg_model: Optional[PriceRegressionModel] = None

        # Track per-model load status and errors
        self._load_errors: Dict[str, str] = {}
        self._loaded_at: Optional[datetime] = None

        # Attempt to load all models eagerly at instantiation
        self._load_all()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Return the process-wide singleton instance.

        Thread-safe: uses a double-checked lock to avoid redundant
        instantiation under concurrent access.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Initialising ModelRegistry singleton.")
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Model accessors
    # ------------------------------------------------------------------

    def get_negative_classifier(self) -> NegativePriceClassifier:
        """Return the loaded NegativePriceClassifier.

        Raises
        ------
        RuntimeError
            If the model was not loaded successfully.
        """
        if self._neg_clf is None:
            err = self._load_errors.get("negative_price_classifier", "unknown error")
            raise RuntimeError(
                f"NegativePriceClassifier is not available: {err}"
            )
        return self._neg_clf

    def get_rebound_classifier(self) -> ReboundClassifier:
        """Return the loaded ReboundClassifier.

        Raises
        ------
        RuntimeError
            If the model was not loaded successfully.
        """
        if self._reb_clf is None:
            err = self._load_errors.get("rebound_classifier", "unknown error")
            raise RuntimeError(
                f"ReboundClassifier is not available: {err}"
            )
        return self._reb_clf

    def get_price_regression(self) -> PriceRegressionModel:
        """Return the loaded PriceRegressionModel.

        Raises
        ------
        RuntimeError
            If the model was not loaded successfully.
        """
        if self._reg_model is None:
            err = self._load_errors.get("price_regression", "unknown error")
            raise RuntimeError(
                f"PriceRegressionModel is not available: {err}"
            )
        return self._reg_model

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def are_models_ready(self) -> bool:
        """Return ``True`` if all three models loaded without errors."""
        return (
            self._neg_clf is not None
            and self._reb_clf is not None
            and self._reg_model is not None
        )

    def get_model_info(self) -> Dict:
        """Return a status / metadata snapshot for all three models.

        Returns
        -------
        dict
            Keys:
            - ``ready`` (bool): all models loaded
            - ``loaded_at`` (str | None): ISO timestamp of last successful load
            - ``model_dir`` (str)
            - ``negative_price_classifier``: metrics + status
            - ``rebound_classifier``: metrics + status
            - ``price_regression``: metrics + status
            - ``load_errors`` (dict): per-model error messages, if any
        """
        def _model_info(model, name: str) -> Dict:
            if model is None:
                return {
                    "loaded": False,
                    "error": self._load_errors.get(name, "not loaded"),
                }
            return {
                "loaded": True,
                "metrics": model.metrics,
                "feature_count": (
                    len(model.feature_names) if model.feature_names else 0
                ),
            }

        training_metrics = self._read_training_metrics()
        last_trained = self._read_last_trained()

        return {
            "ready": self.are_models_ready(),
            "loaded_at": (
                self._loaded_at.isoformat() if self._loaded_at else None
            ),
            "last_trained_at": last_trained,
            "model_dir": self._model_dir,
            "negative_price_classifier": _model_info(
                self._neg_clf, "negative_price_classifier"
            ),
            "rebound_classifier": _model_info(
                self._reb_clf, "rebound_classifier"
            ),
            "price_regression": _model_info(
                self._reg_model, "price_regression"
            ),
            "training_metrics": training_metrics,
            "load_errors": self._load_errors,
        }

    def reload(self) -> bool:
        """Force a reload of all models from disk.

        Useful after a training run to pick up freshly saved artefacts
        without restarting the process.

        Returns
        -------
        bool
            ``True`` if all models loaded successfully after the reload.
        """
        logger.info("ModelRegistry: reloading all models from disk.")
        self._neg_clf = None
        self._reb_clf = None
        self._reg_model = None
        self._load_errors.clear()
        self._load_all()
        return self.are_models_ready()

    # ------------------------------------------------------------------
    # Internal loading logic
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Attempt to load all three models; log but do not raise on failure."""
        # NegativePriceClassifier
        try:
            clf = NegativePriceClassifier(model_dir=self._model_dir)
            clf.load()
            self._neg_clf = clf
            logger.info("NegativePriceClassifier loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to load NegativePriceClassifier: %s", exc
            )
            self._load_errors["negative_price_classifier"] = str(exc)

        # ReboundClassifier
        try:
            reb = ReboundClassifier(model_dir=self._model_dir)
            reb.load()
            self._reb_clf = reb
            logger.info("ReboundClassifier loaded successfully.")
        except Exception as exc:
            logger.warning("Failed to load ReboundClassifier: %s", exc)
            self._load_errors["rebound_classifier"] = str(exc)

        # PriceRegressionModel
        try:
            reg = PriceRegressionModel(model_dir=self._model_dir)
            reg.load()
            self._reg_model = reg
            logger.info("PriceRegressionModel loaded successfully.")
        except Exception as exc:
            logger.warning("Failed to load PriceRegressionModel: %s", exc)
            self._load_errors["price_regression"] = str(exc)

        if self.are_models_ready():
            self._loaded_at = datetime.now(tz=timezone.utc)
            logger.info("All models loaded. Registry is ready.")
        else:
            missing = [
                name
                for name, err in self._load_errors.items()
                if err
            ]
            logger.warning(
                "ModelRegistry partially loaded. Missing: %s",
                ", ".join(missing),
            )

    def _read_training_metrics(self) -> Optional[Dict]:
        """Read persisted training metrics file if present."""
        path = os.path.join(self._model_dir, _METRICS_FILE)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Could not read training metrics: %s", exc)
            return None

    def _read_last_trained(self) -> Optional[str]:
        """Return the ISO timestamp string of the last training run."""
        path = os.path.join(self._model_dir, _LAST_TRAINED_FILE)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                data = json.load(fh)
            return data.get("trained_at")
        except Exception as exc:
            logger.warning("Could not read last_trained.json: %s", exc)
            return None
