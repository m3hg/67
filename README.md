# 2k Vision V2.0 — NBA 2K26 Computer Vision Overlay

A real-time CV overlay tool that draws directly on top of NBA 2K26 gameplay.
It detects the shot meter, tracks the ball trajectory, predicts the optimal
release point, and can auto-fire via a virtual controller.

---

## Features

| Feature | Description |
|---|---|
| **Transparent overlay** | Frameless, click-through window drawn over the game |
| **Shot meter detection** | HSV color thresholding — detects the magenta fill bar |
| **Ball trajectory arc** | Kalman-tracked orange dots forming a parabolic arc |
| **Player bounding boxes** | MOG2 background subtraction detects moving players |
| **Green release flash** | Detects perfect-release green burst near the basket |
| **EMA v9 predictor** | 4-zone graduated fire system with confidence scoring |
| **Adaptive calibration** | Adjusts fire threshold based on post-shot feedback |
| **Controller proxy** | XInput → ViGEm virtual X360 pad; auto-releases shot |
| **YOLO training pipeline** | Capture dataset, train YOLOv8, export to ONNX |
| **Dark UI control panel** | Setup / Tune / Run / Trainer tabs |

---

## Project Structure

```
2k_vision/
├── main.py                  # Entry point
├── config.py                # JSON config + persistence
├── capture.py               # Screen capture (mss / dxcam)
├── overlay.py               # Win32 transparent overlay (PyQt6)
├── detector_meter.py        # Shot meter HSV detector
├── detector_ball.py         # Basketball detector + Kalman tracker
├── detector_player.py       # Player detector (MOG2)
├── detector_green.py        # Green release flash detector
├── predictor.py             # EMA velocity + 4-zone fire system
├── tracker_fps.py           # FPS tracker
├── controller.py            # XInput reader + ViGEm proxy
├── feedback.py              # Post-shot feedback scanner
├── calibrator.py            # Adaptive fire_pct calibration
├── pipeline.py              # Detection pipeline orchestrator
├── engine.py                # State machine (IDLE→SHOOTING→FIRED…)
├── utils.py                 # Shared CV/geometry helpers
├── model_trainer/
│   ├── dataset_builder.py   # Capture + auto-label screenshots
│   ├── train.py             # YOLOv8 training
│   ├── export.py            # Export to ONNX
│   └── inference.py         # ONNX inference runner
└── ui/
    ├── app.py               # Control panel main window
    ├── theme.py             # Dark stylesheet
    ├── tab_setup.py         # System info tab
    ├── tab_tune.py          # Color/detection tuning tab
    ├── tab_run.py           # ARM/DISARM + live stats tab
    ├── tab_trainer.py       # YOLO training tab
    └── widgets.py           # Reusable UI components
```

---

## Requirements

- Windows 10/11 (overlay click-through requires Win32 API)
- Python 3.10+
- NBA 2K26 running in windowed or borderless-windowed mode
- Xbox controller (optional — for auto-fire via ViGEm)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dxcam` and `pywin32` are Windows-only. On other platforms
> only the `mss` capture backend is available and the overlay will run
> without click-through.

### 2. Install ViGEm Bus Driver (for auto-fire)

Download and install from:
https://github.com/ViGEm/ViGEmBus/releases

### 3. Run

```bash
# From the repository root:
python -m 2k_vision.main

# Or:
python 2k_vision/main.py
```

---

## Usage

### Control Panel

| Tab | Purpose |
|---|---|
| **Setup** | System info, dependency status, feature list |
| **Tune** | Adjust shot-meter color, ball color, predictor thresholds |
| **Run** | ARM/DISARM auto-fire, live stats, shot log |
| **Trainer** | Capture dataset, train YOLOv8 model |

### Hotkeys

| Key | Action |
|---|---|
| `F6` | ARM auto-fire |
| `F7` | DISARM |
| `F8` | Toggle overlay visibility |

---

## Shot Meter Tuning

If the shot meter color in your game looks different, use the **Tune** tab
to adjust the HSV values.  The default is magenta (H: 140-170).

For a yellow meter: H 18-35, S 80-255, V 80-255  
For a white meter:  H 0-179, S 0-50, V 200-255

---

## Model Training (optional)

1. Go to the **Trainer** tab and click **Start Capture** while playing.
2. Click **Stop Capture** once you have ≥500 samples.
3. Click **Train YOLOv8** and wait for training to complete.
4. The best weights and ONNX export will be saved automatically.

---

## Configuration

All settings are saved to `~/2k_vision_data/config.json`.  
Shot history is saved to `~/2k_vision_data/history.json`.

---

## License

MIT — for educational / personal use only.  
Not affiliated with 2K Games or Take-Two Interactive.
