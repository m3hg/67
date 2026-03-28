"""
config.py — All configuration + JSON-based persistence for 2k Vision.

Manages detection parameters, overlay toggles, and shot history.
All values are configurable and saved to disk automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.expanduser("~"), "2k_vision_data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MeterConfig:
    """HSV parameters for shot-meter detection (default: magenta)."""

    hue_low: int = 140
    hue_high: int = 170
    sat_low: int = 80
    sat_high: int = 255
    val_low: int = 80
    val_high: int = 255
    # Contour geometry filters
    min_height: int = 15
    max_height: int = 200
    min_width: int = 4
    max_width: int = 60
    min_area: int = 40
    aspect_ratio_min: float = 1.5  # height/width — tall & narrow
    # Morphological kernel size (px)
    morph_kernel: int = 3
    # Reference height for fill percentage (0 = auto-learn from max observed)
    ref_height: int = 0


@dataclass
class BallConfig:
    """HSV parameters and tracking settings for basketball detection."""

    hue_low: int = 5
    hue_high: int = 25
    sat_low: int = 100
    sat_high: int = 255
    val_low: int = 100
    val_high: int = 255
    min_radius: int = 6
    max_radius: int = 60
    circularity_threshold: float = 0.4
    history_len: int = 30
    kalman_process_noise: float = 1e-3
    kalman_meas_noise: float = 1e-1


@dataclass
class PlayerConfig:
    """Settings for player / motion detection."""

    enabled: bool = True
    run_every_n_frames: int = 3
    min_contour_area: int = 800
    max_contour_area: int = 60_000
    min_aspect_ratio: float = 0.5   # height / width must be > this
    skin_detect: bool = False
    bg_history: int = 200
    bg_var_threshold: float = 40.0
    bg_detect_shadows: bool = False


@dataclass
class GreenConfig:
    """HSV parameters for green-release flash detection."""

    hue_low: int = 35
    hue_high: int = 85
    sat_low: int = 80
    sat_high: int = 255
    val_low: int = 180
    val_high: int = 255
    screen_top_fraction: float = 0.6  # only scan top X% of frame
    flash_duration_frames: int = 15
    min_pixel_area: int = 100


@dataclass
class PredictorConfig:
    """EMA velocity predictor settings."""

    ema_alpha: float = 0.3
    outlier_sigma: float = 3.0
    fire_pct: float = 0.88           # default fire threshold (0-1)
    buffer_px: int = 4               # last-chance buffer
    # Zone thresholds
    z2_conf: float = 0.80
    z2_fill: float = 0.85
    z3_conf: float = 0.60
    z3_fill: float = 0.92
    # Lead-time fractions
    z2_lead_frac: float = 0.70
    z3_lead_frac: float = 0.30
    overshoot_guard: float = 1.15    # reject if predicted fill > 115%


@dataclass
class CaptureConfig:
    """Screen capture settings."""

    backend: str = "mss"   # "mss" or "dxcam"
    monitor_index: int = 1
    # Optional manual crop override (null = fullscreen)
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0  # 0 = use full width
    crop_h: int = 0  # 0 = use full height
    target_fps: int = 60


@dataclass
class ControllerConfig:
    """XInput / ViGEm controller proxy settings."""

    enabled: bool = True
    mode: str = "X"           # "X" or "RS"
    xinput_poll_hz: int = 1000
    remap_lb_to_x: bool = True


@dataclass
class OverlayConfig:
    """Overlay element visibility toggles."""

    show_scan_lines: bool = True
    show_meter_box: bool = True
    show_meter_fill: bool = True
    show_fire_line: bool = True
    show_player_boxes: bool = True
    show_ball_trail: bool = True
    show_predicted_arc: bool = True
    show_stats_text: bool = True
    show_timing_hud: bool = True
    show_green_flash: bool = True
    show_fire_indicator: bool = True
    show_watermark: bool = True
    opacity: float = 1.0


@dataclass
class EngineConfig:
    """State-machine timing settings (ms)."""

    start_delay_ms: int = 0
    timeout_ms: int = 4000
    cooldown_ms: int = 400
    scan_duration_ms: int = 600


@dataclass
class TrainerConfig:
    """Dataset capture + YOLO training settings."""

    save_dir: str = os.path.join(DATA_DIR, "dataset")
    capture_interval_ms: int = 500
    capture_hotkey: str = "F9"
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    model_variant: str = "yolov8n"
    classes: list = field(default_factory=lambda: ["ball", "player", "shot_meter", "basket"])


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """Root configuration object — holds all sub-configs."""

    meter: MeterConfig = field(default_factory=MeterConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    green: GreenConfig = field(default_factory=GreenConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _from_dict(cls, data: dict) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    import typing

    if not dataclasses.is_dataclass(cls):
        return data
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict = {}
    for fname, fobj in known.items():
        if fname not in data:
            continue
        val = data[fname]
        ftype = fobj.type
        # Resolve string annotations safely (no eval)
        if isinstance(ftype, str):
            hints = typing.get_type_hints(cls)
            ftype = hints.get(fname, ftype)
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[fname] = _from_dict(ftype, val)
        else:
            kwargs[fname] = val
    return cls(**kwargs)


def load_config() -> AppConfig:
    """Load config from disk; fall back to defaults for missing keys."""
    _ensure_data_dir()
    cfg = AppConfig()
    if not os.path.exists(CONFIG_PATH):
        return cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        defaults = asdict(cfg)
        merged = _deep_update(defaults, raw)
        cfg = _from_dict(AppConfig, merged)
    except Exception:  # pylint: disable=broad-except
        pass  # Return defaults on any parse error
    return cfg


def save_config(cfg: AppConfig) -> None:
    """Persist config to disk."""
    _ensure_data_dir()
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)


# ---------------------------------------------------------------------------
# Shot history helpers
# ---------------------------------------------------------------------------


def load_history() -> list[dict]:
    """Load shot history from disk."""
    _ensure_data_dir()
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # pylint: disable=broad-except
        return []


def save_history(history: list[dict]) -> None:
    """Persist shot history to disk (keep last 1000 entries)."""
    _ensure_data_dir()
    history = history[-1000:]
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the global config singleton, loading from disk on first call."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Force reload from disk and return updated singleton."""
    global _config  # noqa: PLW0603
    _config = load_config()
    return _config
