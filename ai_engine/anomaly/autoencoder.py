"""
Anomaly Detection Module — Phase 8
Provides AnomalyDetector for detecting unusual pose patterns.

Approach
--------
Autoencoder-based anomaly detection using scikit-learn's MLPRegressor as a
lightweight autoencoder (input == output reconstruction). The model is trained
on pose keypoint vectors extracted from normal school corridor footage.

Anomaly score = normalised reconstruction error (MSE) in [0, 1].

Design decisions
----------------
- Uses scikit-learn only (already in requirements.txt). No PyTorch required.
- Input: flattened pose keypoint vector (17 keypoints × 2 coords + 17 confs = 51 dims).
  Low-confidence keypoints are zeroed so the model learns valid poses only.
- Training must be done separately via scripts/train_anomaly.py (not provided yet).
- When no trained model exists, all frames return anomaly_score = 0.0 (graceful fallback).
- Model is saved/loaded via joblib for efficiency.

Limitations
-----------
- Model performance depends entirely on training data quality.
- Do not claim the model is trained unless training was actually performed.
- A freshly instantiated model in ``UNTRAINED`` state returns a score of 0.0.
- 2D projection limitations apply (same as PoseAnalyzer).

Training instructions
---------------------
1. Collect pose data from your camera under normal conditions.
2. Run: python scripts/train_anomaly.py --data-dir data/poses/ --output models/anomaly/
3. The trained model will be saved to models/anomaly/anomaly_detector.joblib.
"""

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn import — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. AnomalyDetector will use fallback mode.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KP_DIM = 17 * 2   # x,y for 17 COCO keypoints  = 34
_CONF_DIM = 17      # confidence scores           = 17
INPUT_DIM = _KP_DIM + _CONF_DIM  # = 51


class ModelState(str, Enum):
    UNTRAINED = "untrained"
    TRAINED   = "trained"
    LOADED    = "loaded"


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Pose-based anomaly detector using autoencoder reconstruction error.

    State machine:
        UNTRAINED → (train) → TRAINED → (save/load) → LOADED

    Usage — inference only (no training data)
    -----------------------------------------
    ::

        detector = AnomalyDetector()
        ok = detector.load_model()   # returns False if no model file found
        if ok:
            score = detector.score(kp_xy, kp_conf)
            if detector.is_anomaly(score):
                print("Unusual pose detected")

    Usage — training
    ----------------
    ::

        detector = AnomalyDetector()
        detector.fit(list_of_kp_xy, list_of_kp_conf)
        detector.save_model()
    """

    _DEFAULT_MODEL_PATH = "models/anomaly/anomaly_detector.joblib"
    _DEFAULT_SCALER_PATH = "models/anomaly/anomaly_scaler.joblib"

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "anomaly_threshold":        0.75,  # normalised score above which is anomaly
        "hidden_layer_sizes":      (64, 16, 64),  # bottleneck architecture
        "max_iter":                500,
        "random_state":            42,
        "confidence_threshold":    0.5,    # keypoint conf below this → zeroed in input
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> None:
        """
        Initialise AnomalyDetector.

        Args:
            config      : Optional config overrides.
            model_path  : Path to saved model joblib file.
            scaler_path : Path to saved scaler joblib file.
        """
        self._cfg = dict(self._DEFAULT_CONFIG)
        if config:
            self._cfg.update(config)

        project_root = Path(__file__).parent.parent.parent
        self._model_path  = Path(model_path  or project_root / self._DEFAULT_MODEL_PATH)
        self._scaler_path = Path(scaler_path or project_root / self._DEFAULT_SCALER_PATH)

        self._model:  Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._state = ModelState.UNTRAINED

        # For normalising reconstruction error: estimated from training data
        self._max_reconstruction_error: float = 1.0

        logger.info("AnomalyDetector initialised (state=%s)", self._state.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("Cannot load model: scikit-learn not installed.")
            return False

        try:
            if not self._model_path.exists():
                logger.info("No trained model found at %s — running in UNTRAINED mode.",
                            self._model_path)
                return False

            self._model  = joblib.load(str(self._model_path))
            if self._scaler_path.exists():
                self._scaler = joblib.load(str(self._scaler_path))

            # Load metadata if present
            meta_path = self._model_path.with_suffix(".meta.pkl")
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                self._max_reconstruction_error = meta.get("max_reconstruction_error", 1.0)

            self._state = ModelState.LOADED
            logger.info("AnomalyDetector model loaded from %s", self._model_path)
            return True

        except Exception as exc:
            logger.error("Failed to load anomaly model: %s", exc)
            return False

    def fit(
        self,
        keypoints_xy_list: List[np.ndarray],
        keypoints_conf_list: List[np.ndarray],
    ) -> "AnomalyDetector":
        """
        Train the autoencoder on a collection of normal-activity pose samples.

        Args:
            keypoints_xy_list   : List of (17, 2) keypoint coordinate arrays.
            keypoints_conf_list : List of (17,) keypoint confidence arrays.

        Returns:
            self (for chaining)

        Raises:
            RuntimeError: If scikit-learn is not available.
            ValueError:   If input lists are empty.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training. Install it with: pip install scikit-learn")

        if not keypoints_xy_list:
            raise ValueError("No training samples provided.")

        logger.info("Fitting AnomalyDetector on %d samples...", len(keypoints_xy_list))

        # Build feature matrix
        X = np.array([
            self._encode(kp_xy, kp_conf)
            for kp_xy, kp_conf in zip(keypoints_xy_list, keypoints_conf_list)
        ])

        # Normalise features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Build and train autoencoder
        self._model = MLPRegressor(
            hidden_layer_sizes=tuple(self._cfg["hidden_layer_sizes"]),
            max_iter=self._cfg["max_iter"],
            random_state=self._cfg["random_state"],
            verbose=False,
        )
        self._model.fit(X_scaled, X_scaled)  # autoencoder: reconstruct input

        # Compute max reconstruction error on training data for normalisation
        X_pred = self._model.predict(X_scaled)
        errors = np.mean((X_scaled - X_pred) ** 2, axis=1)
        self._max_reconstruction_error = float(np.percentile(errors, 99)) or 1.0

        self._state = ModelState.TRAINED
        logger.info("Training complete. Max 99th-pct reconstruction error: %.6f",
                    self._max_reconstruction_error)
        return self

    def save_model(self) -> bool:
        """
        Save trained model and scaler to disk.

        Returns:
            True on success, False on failure.
        """
        if self._state == ModelState.UNTRAINED:
            logger.warning("Cannot save untrained model.")
            return False
        if not _SKLEARN_AVAILABLE:
            return False

        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model,  str(self._model_path))
            if self._scaler is not None:
                joblib.dump(self._scaler, str(self._scaler_path))
            # Save metadata
            meta_path = self._model_path.with_suffix(".meta.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump({"max_reconstruction_error": self._max_reconstruction_error}, f)
            logger.info("Model saved to %s", self._model_path)
            return True
        except Exception as exc:
            logger.error("Failed to save model: %s", exc)
            return False

    def score(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
    ) -> float:
        """
        Compute normalised anomaly score for a single person's pose.

        Returns:
            Float in [0.0, 1.0]. Higher = more anomalous.
            Returns 0.0 when model is untrained or sklearn unavailable.
        """
        if self._state == ModelState.UNTRAINED or self._model is None:
            return 0.0
        if not _SKLEARN_AVAILABLE:
            return 0.0

        try:
            vec = self._encode(keypoints_xy, keypoints_conf).reshape(1, -1)
            if self._scaler is not None:
                vec = self._scaler.transform(vec)
            pred = self._model.predict(vec)
            mse  = float(np.mean((vec - pred) ** 2))
            # Normalise to [0, 1] using training max error
            norm_score = min(1.0, mse / max(self._max_reconstruction_error, 1e-9))
            return round(norm_score, 6)
        except Exception as exc:
            logger.debug("Score computation error: %s", exc)
            return 0.0

    def is_anomaly(self, score: float) -> bool:
        """Return True if score exceeds the configured anomaly threshold."""
        return score >= float(self._cfg["anomaly_threshold"])

    def get_state(self) -> str:
        """Return current model state string."""
        return self._state.value

    def get_config(self) -> Dict[str, Any]:
        """Return copy of current configuration."""
        return dict(self._cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> np.ndarray:
        """
        Flatten pose keypoints into a fixed-dimension feature vector.

        Low-confidence keypoints are zeroed so the model is not confused by
        pose estimator noise. Output is always INPUT_DIM (51) values.
        """
        threshold  = float(self._cfg["confidence_threshold"])
        kp_xy_arr  = np.asarray(kp_xy,  dtype=np.float32).reshape(-1, 2)[:17]
        kp_conf_arr= np.asarray(kp_conf, dtype=np.float32).flatten()[:17]

        # Pad if fewer than 17 keypoints
        if len(kp_xy_arr) < 17:
            pad = np.zeros((17 - len(kp_xy_arr), 2), dtype=np.float32)
            kp_xy_arr = np.vstack([kp_xy_arr, pad])
        if len(kp_conf_arr) < 17:
            kp_conf_arr = np.pad(kp_conf_arr, (0, 17 - len(kp_conf_arr)))

        # Zero out low-confidence keypoints
        mask = kp_conf_arr < threshold
        kp_xy_arr_masked = kp_xy_arr.copy()
        kp_xy_arr_masked[mask] = 0.0

        # Flatten: 34 xy coords + 17 confidence values
        vec = np.concatenate([kp_xy_arr_masked.flatten(), kp_conf_arr])
        return vec.astype(np.float32)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AnomalyDetector(state={self._state.value}, threshold={self._cfg['anomaly_threshold']})"
