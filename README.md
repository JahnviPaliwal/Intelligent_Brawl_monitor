# Intelligent Brawl Monitor v3.0

Real-time fight detection using **MediaPipe Pose** in the browser, with an automated email alert pipeline.

---

## How it works

1. **Camera feed** — Your webcam stream is processed frame-by-frame in the browser using **TensorFlow.js MoveNet MultiPose Lightning** — tracks up to 6 persons simultaneously with 17 keypoints each. No data leaves your device.
2. **Multi-person pose analysis** — For every person in frame, the system runs a rule-based classifier over their keypoints:
   - **Punch** — one wrist above shoulder, elbow extended ≥140°, fast wrist velocity
   - **Kick** — knee raised above hip, leg extended ≥120°
   - **Guard** — both wrists at face level, elbows bent 60–130°, arms close together
   - **Grab / Clinch** — arms spread wide at shoulder height
   - **Neutral** — no aggressive indicators
3. **Threat scoring** — Each person gets a base threat score by pose class, boosted by wrist velocity. Multi-person scenes with 2+ aggressive poses get a stacking bonus.
4. **Alert dispatch** — When combined threat exceeds threshold for N consecutive frames, an alert fires (email via backend, beep sound, on-screen log).

---

## Quick start (no backend needed)

Just open `index.html` in Chrome or Edge (Firefox may block camera on `file://`).

- Click **Start camera** — uses your webcam with real MediaPipe inference
- Click **Simulate fight** — runs a synthetic skeleton demo without any camera

> If MediaPipe CDN fails (no internet), it automatically falls back to simulation mode.

---

## With Python backend (real email alerts)

The backend serves the app and handles email dispatch.

### 1. Install (no pip packages needed — standard library only)

```bash
cd backend
python server.py
```

### 2. Configure email

Set environment variables before running:

```bash
# Linux / macOS
export SENDER_EMAIL="you@gmail.com"
export SENDER_PASS="your-gmail-app-password"
export RECIPIENT_EMAIL="security@yourplace.com"
python backend/server.py

# Windows
set SENDER_EMAIL=you@gmail.com
set SENDER_PASS=your-gmail-app-password
set RECIPIENT_EMAIL=security@yourplace.com
python backend\server.py
```

Or fill them in the **Settings** tab of the UI — they are sent with each alert POST request.

### 3. Gmail App Password

Gmail requires an **App Password** (not your normal password) when 2FA is enabled:
1. Go to [myaccount.google.com/security](https://myaccount.google.com/security)
2. Under "Signing in to Google" → "App passwords"
3. Create a password for "Mail" → copy the 16-character code

### 4. Open the app

Visit [http://localhost:5000](http://localhost:5000)

---

## Project structure

```
brawl_monitor/
├── index.html          ← Main app (open this directly, or via server)
├── css/
│   └── style.css       ← Dark dashboard UI
├── js/
│   └── app.js          ← MediaPipe integration + fight logic + alert dispatch
├── backend/
│   └── server.py       ← Python HTTP server + email (stdlib only, no pip needed)
└── README.md
```

---

## What changed from v2 → v3

| v2 issue | Fix in v3 |
|---|---|
| MediaPipe Pose — single person only | TF.js MoveNet MultiPose — up to 6 persons simultaneously |
| Only "wrists up + elbow angle" heuristic | Full pose classifier: Punch / Kick / Guard / Grab / Neutral |
| No per-person labels on canvas | Each person gets bounding box + pose label + colour-coded skeleton |
| Multi-person was faked with simulation overlay | Real multi-person keypoints, each analysed independently |
| Threat score only from single person | Per-person scores with interaction bonuses when 2+ are aggressive |

---

## Settings (in UI)

| Setting | Default | Description |
|---|---|---|
| Threat threshold | 65% | Minimum score to trigger alert |
| Frames required | 15 | Consecutive frames before alert fires (~0.5s at 30fps) |
| Wrist elevation margin | 0.05 | How far above shoulder wrists must be |
| Show skeleton | On | Overlay pose skeleton on video |
| Mirror camera | On | Flip video horizontally |
| Alert sound | On | Play beep on fight detection |

---

## Limitations

- MediaPipe Pose detects **one person** per model instance. Multi-person detection in this build uses a secondary synthetic overlay for demo; a production deployment would run parallel instances or use a multi-person model.
- Performance depends on device GPU. Chrome/Edge use WebGL acceleration.
- Camera access requires HTTPS or `localhost`.
