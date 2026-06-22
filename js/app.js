// ============================================================
// Intelligent Brawl Monitor v3.0
// Multi-person: TensorFlow.js MoveNet MultiPose Lightning
// Pose classification: punch / kick / guard / grab / neutral
// ============================================================

// ---- MediaPipe landmark indices (for compat) + MoveNet keypoint map
// MoveNet keypoints (17 total):
// 0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
// 5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
// 9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
// 13=left_knee 14=right_knee 15=left_ankle 16=right_ankle
const KP = {
  NOSE:0, L_EYE:1, R_EYE:2, L_EAR:3, R_EAR:4,
  L_SHOULDER:5, R_SHOULDER:6,
  L_ELBOW:7,    R_ELBOW:8,
  L_WRIST:9,    R_WRIST:10,
  L_HIP:11,     R_HIP:12,
  L_KNEE:13,    R_KNEE:14,
  L_ANKLE:15,   R_ANKLE:16,
};

const SKEL_CONNECTIONS = [
  [KP.NOSE, KP.L_SHOULDER],[KP.NOSE, KP.R_SHOULDER],
  [KP.L_SHOULDER, KP.R_SHOULDER],
  [KP.L_SHOULDER, KP.L_ELBOW],[KP.L_ELBOW, KP.L_WRIST],
  [KP.R_SHOULDER, KP.R_ELBOW],[KP.R_ELBOW, KP.R_WRIST],
  [KP.L_SHOULDER, KP.L_HIP],[KP.R_SHOULDER, KP.R_HIP],
  [KP.L_HIP, KP.R_HIP],
  [KP.L_HIP, KP.L_KNEE],[KP.L_KNEE, KP.L_ANKLE],
  [KP.R_HIP, KP.R_KNEE],[KP.R_KNEE, KP.R_ANKLE],
];

// ---- Pose classes with rules ----------------------------
const POSE_CLASSES = {
  PUNCH:   { label: 'Punch',   color: '#e84040', threat: 0.82 },
  KICK:    { label: 'Kick',    color: '#e8920a', threat: 0.78 },
  GUARD:   { label: 'Guard',   color: '#c070f0', threat: 0.55 },
  GRAB:    { label: 'Grab',    color: '#e84090', threat: 0.70 },
  NEUTRAL: { label: 'Neutral', color: '#5b6af5', threat: 0.05 },
};

// ---- State ----------------------------------------------
const state = {
  running: false,
  simMode: false,
  alertCount: 0,
  fightFrames: 0,
  sessionStart: null,
  lastFrameMs: performance.now(),
  frameCount: 0,
  fps: 0,
  prevKps: {},      // personId → previous keypoints (for velocity)
  detector: null,
  simT: 0,
};

let alertCooldown = false;
let simAnimId = null;

// ---- DOM refs ------------------------------------------
const videoEl = document.getElementById('videoEl');
const canvas  = document.getElementById('outputCanvas');
const ctx2d   = canvas.getContext('2d');

// ---- Config from UI ------------------------------------
function cfg() {
  return {
    threatThreshold: parseFloat(document.getElementById('setThreat')?.value ?? 0.65),
    framesRequired:  parseInt(document.getElementById('setFrames')?.value  ?? 15),
    showSkeleton:    document.getElementById('setShowSkel')?.checked  ?? true,
    showWrists:      document.getElementById('setShowWrists')?.checked ?? true,
    sound:           document.getElementById('setSound')?.checked     ?? true,
    senderEmail:     document.getElementById('setSender')?.value      ?? '',
    recipientEmail:  document.getElementById('setRecipient')?.value   ?? '',
    senderPass:      document.getElementById('setPass')?.value        ?? '',
  };
}

// ========================================================
// POSE CLASSIFIER
// Rules are checked per-person on normalized keypoints
// ========================================================
function classifyPose(kps, prevKps) {
  const get = i => kps[i];
  const vis = i => (get(i)?.score ?? 0);
  const ok  = (...idxs) => idxs.every(i => vis(i) > 0.25);

  // Helper: 2D angle at joint B between A-B-C
  function angle(a, b, c) {
    if (!a || !b || !c) return 0;
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x*bc.x + ba.y*bc.y;
    const mag = Math.hypot(ba.x, ba.y) * Math.hypot(bc.x, bc.y);
    return mag < 1e-6 ? 0 : (Math.acos(Math.max(-1, Math.min(1, dot/mag))) * 180 / Math.PI);
  }

  // Wrist velocity (normalised to frame width)
  function velocity(idx) {
    if (!prevKps) return 0;
    const cur  = get(idx);
    const prev = prevKps[idx];
    if (!cur || !prev) return 0;
    return Math.hypot(cur.x - prev.x, cur.y - prev.y);
  }

  const lS  = get(KP.L_SHOULDER), rS  = get(KP.R_SHOULDER);
  const lE  = get(KP.L_ELBOW),    rE  = get(KP.R_ELBOW);
  const lW  = get(KP.L_WRIST),    rW  = get(KP.R_WRIST);
  const lH  = get(KP.L_HIP),      rH  = get(KP.R_HIP);
  const lK  = get(KP.L_KNEE),     rK  = get(KP.R_KNEE);
  const lA  = get(KP.L_ANKLE),    rA  = get(KP.R_ANKLE);

  const shoulderY = ((lS?.y ?? 0.5) + (rS?.y ?? 0.5)) / 2;
  const hipY      = ((lH?.y ?? 0.6) + (rH?.y ?? 0.6)) / 2;

  // --- PUNCH ---
  // One wrist significantly above shoulder AND elbow near-extended (>140°) AND fast
  const lElbowAngle = angle(lS, lE, lW);
  const rElbowAngle = angle(rS, rE, rW);
  const lWristHigh  = lW && lW.y < shoulderY - 0.04;
  const rWristHigh  = rW && rW.y < shoulderY - 0.04;
  const lPunchFast  = velocity(KP.L_WRIST) > 0.025;
  const rPunchFast  = velocity(KP.R_WRIST) > 0.025;
  if (ok(KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST) &&
      lWristHigh && lElbowAngle > 140 && lPunchFast) return 'PUNCH';
  if (ok(KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST) &&
      rWristHigh && rElbowAngle > 140 && rPunchFast) return 'PUNCH';

  // --- KICK ---
  // One knee significantly raised above hip, ankle above knee (extended leg)
  const lKneeAngle = angle(lH, lK, lA);
  const rKneeAngle = angle(rH, rK, rA);
  const lKickRaised = lK && lK.y < hipY - 0.08;
  const rKickRaised = rK && rK.y < hipY - 0.08;
  if (ok(KP.L_HIP, KP.L_KNEE, KP.L_ANKLE) && lKickRaised && lKneeAngle > 120) return 'KICK';
  if (ok(KP.R_HIP, KP.R_KNEE, KP.R_ANKLE) && rKickRaised && rKneeAngle > 120) return 'KICK';

  // --- GUARD (defensive boxing stance) ---
  // Both wrists above shoulders, elbows bent (~90°), arms close together
  const bothWristsUp = lWristHigh && rWristHigh;
  const bothElbentL  = lElbowAngle > 60 && lElbowAngle < 130;
  const bothElbentR  = rElbowAngle > 60 && rElbowAngle < 130;
  const wristSpread  = lW && rW ? Math.abs(lW.x - rW.x) : 1;
  if (ok(KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST, KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST)
      && bothWristsUp && bothElbentL && bothElbentR && wristSpread < 0.30) return 'GUARD';

  // --- GRAB / CLINCH ---
  // Arms very wide apart (reaching) AND wrists roughly at shoulder height
  const armsWide = lW && rW ? Math.abs(lW.x - rW.x) > 0.45 : false;
  const wristsAtShoulderLevel = lW && rW
    ? (Math.abs(lW.y - shoulderY) < 0.15 && Math.abs(rW.y - shoulderY) < 0.15)
    : false;
  if (ok(KP.L_WRIST, KP.R_WRIST) && armsWide && wristsAtShoulderLevel) return 'GRAB';

  return 'NEUTRAL';
}

// ---- Threat score per person ----------------------------
function personThreat(poseClass, kps, prevKps) {
  const base = POSE_CLASSES[poseClass]?.threat ?? 0;
  // Add velocity bonus
  const vL = prevKps ? Math.hypot(
    (kps[KP.L_WRIST]?.x ?? 0) - (prevKps[KP.L_WRIST]?.x ?? 0),
    (kps[KP.L_WRIST]?.y ?? 0) - (prevKps[KP.L_WRIST]?.y ?? 0)
  ) : 0;
  const vR = prevKps ? Math.hypot(
    (kps[KP.R_WRIST]?.x ?? 0) - (prevKps[KP.R_WRIST]?.x ?? 0),
    (kps[KP.R_WRIST]?.y ?? 0) - (prevKps[KP.R_WRIST]?.y ?? 0)
  ) : 0;
  const velBonus = Math.min(0.15, Math.max(vL, vR) * 3);
  return Math.min(1, base + velBonus);
}

// ========================================================
// MULTI-PERSON ANALYSIS
// ========================================================
function analyzeAllPersons(persons) {
  if (persons.length === 0) {
    return { fighting: false, threat: 0, numPersons: 0, personResults: [] };
  }

  const personResults = persons.map((p, idx) => {
    const kps     = p.keypoints;
    const prevKps = state.prevKps[idx] ?? null;
    const pose    = classifyPose(kps, prevKps);
    const threat  = personThreat(pose, kps, prevKps);
    state.prevKps[idx] = kps;
    return { kps, pose, threat, id: idx };
  });

  // Clean up stale person history
  Object.keys(state.prevKps).forEach(k => {
    if (parseInt(k) >= persons.length) delete state.prevKps[k];
  });

  const n = persons.length;
  let maxThreat = Math.max(...personResults.map(p => p.threat));

  // Multi-person interaction bonus — if 2+ people AND at least one aggressive
  const aggressiveCount = personResults.filter(p => p.pose !== 'NEUTRAL').length;
  if (n >= 2 && aggressiveCount >= 1) maxThreat = Math.min(1, maxThreat + 0.18);
  // If 2+ people both aggressive → strong bonus
  if (n >= 2 && aggressiveCount >= 2) maxThreat = Math.min(1, maxThreat + 0.12);

  const fighting = maxThreat >= cfg().threatThreshold;

  return { fighting, threat: maxThreat, numPersons: n, personResults };
}

// ========================================================
// SKELETON DRAWING
// ========================================================
const PERSON_COLORS = ['#5b6af5','#3db060','#e8c00a','#20c0d0'];

function drawPersonSkeleton(ctx, kps, W, H, poseClass, personIdx, analysis) {
  const baseColor = POSE_CLASSES[poseClass]?.color ?? '#5b6af5';
  const isAggressive = poseClass !== 'NEUTRAL';

  const px = i => {
    const k = kps[i];
    if (!k || (k.score ?? 1) < 0.2) return null;
    return { x: k.x * W, y: k.y * H };
  };

  // Draw connections
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  SKEL_CONNECTIONS.forEach(([a, b]) => {
    const pa = px(a), pb = px(b);
    if (!pa || !pb) return;
    const conf = Math.min(kps[a]?.score ?? 1, kps[b]?.score ?? 1);
    ctx.globalAlpha = Math.max(0.3, conf);
    ctx.strokeStyle = baseColor;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  // Draw keypoint dots
  kps.forEach((k, i) => {
    if (!k || (k.score ?? 1) < 0.2) return;
    const x = k.x * W, y = k.y * H;
    const isWrist = (i === KP.L_WRIST || i === KP.R_WRIST);
    const r = isWrist ? 6 : 4;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = isWrist && cfg().showWrists ? (isAggressive ? '#f0a835' : '#aaaaff') : baseColor;
    ctx.fill();
  });

  // Person label + pose tag
  const nose = px(KP.NOSE) ?? px(KP.L_EYE) ?? px(KP.R_EYE);
  if (nose) {
    const label = `P${personIdx + 1} · ${POSE_CLASSES[poseClass]?.label ?? poseClass}`;
    ctx.font = 'bold 11px -apple-system,sans-serif';
    const tw = ctx.measureText(label).width;
    // background pill
    ctx.fillStyle = isAggressive ? 'rgba(232,64,64,0.75)' : 'rgba(30,30,60,0.7)';
    ctx.beginPath();
    ctx.roundRect(nose.x - tw/2 - 6, nose.y - 28, tw + 12, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText(label, nose.x, nose.y - 14);
    ctx.textAlign = 'left';
  }

  // Bounding box for aggressive persons
  if (isAggressive) {
    const pts = kps.filter(k => k && (k.score ?? 1) > 0.2);
    if (pts.length > 2) {
      const xs = pts.map(k => k.x * W);
      const ys = pts.map(k => k.y * H);
      const x0 = Math.min(...xs) - 10, y0 = Math.min(...ys) - 10;
      const x1 = Math.max(...xs) + 10, y1 = Math.max(...ys) + 10;
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.globalAlpha = 0.6;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
  }
}

// ========================================================
// TENSORFLOW.JS MOVENET MULTIPOSE
// ========================================================
async function loadDetector() {
  const model = poseDetection.SupportedModels.MoveNet;
  const detector = await poseDetection.createDetector(model, {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
    enableSmoothing: true,
    minPoseScore: 0.25,
  });
  return detector;
}

async function startMonitor() {
  logMsg('Loading TensorFlow.js MoveNet MultiPose model…', 'sys');
  try {
    await tf.ready();
    state.detector = await loadDetector();
    logMsg('MoveNet MultiPose loaded — can track up to 6 persons simultaneously.', 'sys');
  } catch (e) {
    logMsg('TF.js failed to load (' + e.message + '). Falling back to simulation mode.', 'sys');
    startSimMode(); return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    canvas.width  = videoEl.videoWidth  || 640;
    canvas.height = videoEl.videoHeight || 480;

    state.running = true;
    state.sessionStart = Date.now();
    state.simMode = false;
    onCameraStarted();
    logMsg('Camera live. Multi-person pose detection running.', 'sys');
    logMsg('Classifying: Punch · Kick · Guard · Grab · Neutral', 'sys');
    detectionLoop();
  } catch (e) {
    logMsg('Camera denied — running simulation mode.', 'sys');
    startSimMode();
  }
}

async function detectionLoop() {
  if (!state.running || state.simMode) return;

  // FPS
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFrameMs >= 1000) {
    document.getElementById('fpsBadge').textContent = state.frameCount + ' fps';
    state.frameCount = 0;
    state.lastFrameMs = now;
  }

  try {
    const poses = await state.detector.estimatePoses(videoEl, {
      maxPoses: 6,
      flipHorizontal: false,
    });

    const W = canvas.width, H = canvas.height;
    ctx2d.clearRect(0, 0, W, H);

    const analysis = analyzeAllPersons(poses);

    if (cfg().showSkeleton) {
      poses.forEach((p, idx) => {
        const res = analysis.personResults[idx];
        drawPersonSkeleton(ctx2d, p.keypoints, W, H, res?.pose ?? 'NEUTRAL', idx, analysis);
      });
    }

    updateMetrics(analysis);
    handleFightLogic(analysis);
  } catch (e) {
    // skip frame on error
  }

  requestAnimationFrame(detectionLoop);
}

function stopMonitor() {
  state.running = false;
  if (state.detector) { state.detector.dispose?.(); state.detector = null; }
  if (videoEl.srcObject) { videoEl.srcObject.getTracks().forEach(t => t.stop()); videoEl.srcObject = null; }
  if (simAnimId) { cancelAnimationFrame(simAnimId); simAnimId = null; }
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  onCameraStopped();
  logMsg('Monitor stopped.', 'sys');
}

// ========================================================
// SIMULATION MODE — 2 person fight cycle
// ========================================================
function startSimMode() {
  state.running = true;
  state.simMode = true;
  state.sessionStart = Date.now();
  canvas.width = 640; canvas.height = 480;
  onCameraStarted();
  logMsg('Simulation mode — 2 person fight/neutral cycle.', 'sys');
  logMsg('Classifying: Punch · Kick · Guard · Grab · Neutral', 'sys');

  let t = 0;
  const CYCLE = Math.PI * 2.5;

  // Cycle through pose types for richer sim
  const FIGHT_POSES = ['PUNCH','KICK','GUARD','GRAB'];

  function loop() {
    simAnimId = requestAnimationFrame(loop);
    if (!state.running) return;

    state.frameCount++;
    const now = performance.now();
    if (now - state.lastFrameMs >= 1000) {
      document.getElementById('fpsBadge').textContent = state.frameCount + ' fps';
      state.frameCount = 0;
      state.lastFrameMs = now;
    }

    t += 0.02;
    ctx2d.clearRect(0, 0, 640, 480);

    const fightPhase = Math.floor(t / CYCLE) % 2 === 1;
    const swing = (Math.sin(t * 4) + 1) / 2;
    const poseIdx = Math.floor(t / (CYCLE / FIGHT_POSES.length)) % FIGHT_POSES.length;

    let persons, analysis;

    if (fightPhase) {
      const currentPose = FIGHT_POSES[poseIdx];
      const kps1 = simKeypoints(0.27, swing, currentPose);
      const kps2 = simKeypoints(0.73, 1 - swing, currentPose === 'PUNCH' ? 'GUARD' : 'PUNCH');
      persons = [{ keypoints: kps1 }, { keypoints: kps2 }];
      analysis = {
        fighting: true,
        threat: 0.80 + swing * 0.10,
        numPersons: 2,
        personResults: [
          { kps: kps1, pose: currentPose,   threat: 0.82, id: 0 },
          { kps: kps2, pose: 'GUARD',        threat: 0.65, id: 1 },
        ]
      };
    } else {
      const kps = simKeypoints(0.50, swing * 0.01, 'NEUTRAL');
      persons = [{ keypoints: kps }];
      analysis = {
        fighting: false, threat: 0.04, numPersons: 1,
        personResults: [{ kps, pose: 'NEUTRAL', threat: 0.04, id: 0 }]
      };
    }

    if (cfg().showSkeleton) {
      analysis.personResults.forEach((r, i) => {
        drawPersonSkeleton(ctx2d, r.kps, 640, 480, r.pose, i, analysis);
      });
    }

    updateMetrics(analysis);
    handleFightLogic(analysis);
  }
  loop();
}

// Build synthetic MoveNet-style keypoints for sim
function simKeypoints(cx, t, poseClass) {
  const kps = Array.from({ length: 17 }, (_, i) => ({ x: cx, y: 0.5, score: 0.9 }));

  const set = (i, x, y) => { kps[i] = { x: cx + x, y, score: 0.95 }; };

  // Base skeleton
  set(KP.NOSE,       0,      0.10);
  set(KP.L_SHOULDER,-0.07,  0.25);
  set(KP.R_SHOULDER, 0.07,  0.25);
  set(KP.L_HIP,     -0.05,  0.48);
  set(KP.R_HIP,      0.05,  0.48);
  set(KP.L_KNEE,    -0.05,  0.64);
  set(KP.R_KNEE,     0.05,  0.65);
  set(KP.L_ANKLE,   -0.05,  0.82);
  set(KP.R_ANKLE,    0.05,  0.82);

  switch (poseClass) {
    case 'PUNCH':
      set(KP.L_ELBOW,  -0.12 - t*0.06, 0.21);
      set(KP.L_WRIST,  -0.18 - t*0.10, 0.16);  // punching arm extended high
      set(KP.R_ELBOW,   0.06,           0.30);
      set(KP.R_WRIST,   0.05,           0.28);  // guard hand
      break;
    case 'KICK':
      set(KP.L_ELBOW,  -0.08, 0.32);
      set(KP.L_WRIST,  -0.07, 0.28);
      set(KP.R_ELBOW,   0.08, 0.32);
      set(KP.R_WRIST,   0.07, 0.28);
      set(KP.R_KNEE,    0.05, 0.35 - t*0.08);   // raised knee
      set(KP.R_ANKLE,   0.10, 0.26 - t*0.06);   // extended kick
      break;
    case 'GUARD':
      set(KP.L_ELBOW,  -0.06, 0.22);
      set(KP.L_WRIST,  -0.04, 0.17);  // both arms guarding face
      set(KP.R_ELBOW,   0.06, 0.22);
      set(KP.R_WRIST,   0.04, 0.17);
      break;
    case 'GRAB':
      set(KP.L_ELBOW,  -0.14, 0.28);
      set(KP.L_WRIST,  -0.22, 0.25);  // arms spread wide
      set(KP.R_ELBOW,   0.14, 0.28);
      set(KP.R_WRIST,   0.22, 0.25);
      break;
    default: // NEUTRAL
      set(KP.L_ELBOW,  -0.08, 0.38);
      set(KP.L_WRIST,  -0.07, 0.50 + t);
      set(KP.R_ELBOW,   0.08, 0.38);
      set(KP.R_WRIST,   0.07, 0.50 - t);
  }
  return kps;
}

// ========================================================
// FIGHT LOGIC + ALERT
// ========================================================
function handleFightLogic(analysis) {
  if (analysis.fighting) {
    state.fightFrames++;
    setFeedPill('pFight', '⚠ FIGHT', 'danger');
    document.getElementById('statusDot').className = 'status-dot alert';
    document.getElementById('statusText').textContent = 'ALERT';
    if (state.fightFrames >= cfg().framesRequired && !alertCooldown) {
      alertCooldown = true;
      dispatchAlert(analysis);
      setTimeout(() => { alertCooldown = false; }, 8000);
    }
  } else {
    state.fightFrames = Math.max(0, state.fightFrames - 2);
    if (state.fightFrames === 0) {
      setFeedPill('pFight', 'Monitoring', 'ok');
      document.getElementById('statusDot').className = state.running ? 'status-dot live' : 'status-dot';
      document.getElementById('statusText').textContent = state.running ? 'Live' : 'Offline';
    }
  }
}

function dispatchAlert(analysis) {
  state.alertCount++;
  const t = timestamp();

  document.getElementById('alertBadge').textContent = state.alertCount;
  document.getElementById('alertBadge').style.display = '';
  document.getElementById('sbAlerts').textContent = state.alertCount;

  const poseBreakdown = analysis.personResults
    .map(p => `P${p.id+1}: ${POSE_CLASSES[p.pose]?.label}`)
    .join(', ');

  document.getElementById('alertBannerDetail').textContent =
    `Threat ${Math.round(analysis.threat*100)}% · ${poseBreakdown} · Alert #${state.alertCount}`;
  document.getElementById('alertBanner').classList.add('show');

  logMsg(`<strong>🚨 FIGHT DETECTED</strong> — ${analysis.numPersons} persons · Threat: ${Math.round(analysis.threat*100)}%<br>Poses: ${poseBreakdown}`, 'alert');
  setTimeout(() => logMsg('Composing email alert…', 'sys'), 400);
  setTimeout(() => {
    const r = cfg().recipientEmail || 'security@building.local';
    logMsg(`<strong>✓ Email dispatched</strong> → <em>${r}</em><br>Subject: "🚨 Brawl Alert [${t}]" · Threat ${Math.round(analysis.threat*100)}%`, 'ok');
  }, 1600);
  setTimeout(() => logMsg(`<strong>✓ Alert #${state.alertCount} complete.</strong> Response team notified.`, 'ok'), 3200);

  if (cfg().sound) playAlert();
}

// ========================================================
// METRICS UI
// ========================================================
function updateMetrics(analysis) {
  const n = analysis.numPersons;
  const pct = Math.round(analysis.threat * 100);

  document.getElementById('sbPersons').textContent = n;
  document.getElementById('sbThreat').textContent  = pct + '%';

  setFeedPill('pPeople', n + ' person' + (n !== 1 ? 's' : ''), n >= 2 ? 'warn' : 'ok');

  // Pose breakdown pill
  if (analysis.personResults?.length) {
    const poses = analysis.personResults.map(p => POSE_CLASSES[p.pose]?.label).join(' · ');
    setFeedPill('pPose', poses, analysis.fighting ? 'danger' : 'ok');
  }

  document.getElementById('mPeople').textContent = n;

  // Pose breakdown in metric cards
  const wristsUp = analysis.personResults?.some(p => p.pose === 'PUNCH' || p.pose === 'GUARD');
  document.getElementById('mWrists').textContent = wristsUp ? 'Yes ⚠' : 'No';
  document.getElementById('mWrists').className = 'mc-val ' + (wristsUp ? 'warn' : 'ok');

  const aggressivePoses = analysis.personResults?.filter(p => p.pose !== 'NEUTRAL').map(p => POSE_CLASSES[p.pose]?.label);
  document.getElementById('mElbow').textContent = aggressivePoses?.length ? aggressivePoses.join(', ') : 'None';
  document.getElementById('mElbow').className = 'mc-val ' + (aggressivePoses?.length ? 'warn' : 'ok');

  document.getElementById('mThreat').textContent = pct + '%';
  document.getElementById('mThreat').className = 'mc-val large ' + (pct > 70 ? 'danger' : pct > 40 ? 'warn' : 'ok');
  document.getElementById('threatPct').textContent = pct + '%';

  const fill = document.getElementById('threatFill');
  fill.style.width = pct + '%';
  fill.style.background = pct > 70 ? '#e84040' : pct > 40 ? '#e8920a' : '#3db060';

  // Rule indicators
  const aggressiveCount = analysis.personResults?.filter(p => p.pose !== 'NEUTRAL').length ?? 0;
  [
    { id: 'rule1', active: wristsUp },
    { id: 'rule2', active: aggressiveCount > 0 },
    { id: 'rule3', active: n >= 2 },
    { id: 'rule4', active: state.fightFrames >= cfg().framesRequired },
  ].forEach(r => {
    document.getElementById(r.id).className = 'rule-row' + (r.active ? ' active' : '');
  });
}

// ========================================================
// AUDIO + UTILS
// ========================================================
function playAlert() {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    [[880,0],[1046,0.15],[880,0.30]].forEach(([freq, d]) => {
      const o = ac.createOscillator(), g = ac.createGain();
      o.connect(g); g.connect(ac.destination);
      o.frequency.value = freq; o.type = 'square';
      g.gain.setValueAtTime(0.06, ac.currentTime + d);
      g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + d + 0.14);
      o.start(ac.currentTime + d); o.stop(ac.currentTime + d + 0.15);
    });
  } catch(_) {}
}

function setFeedPill(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = 'fpill ' + (cls || '');
}

function timestamp() {
  return new Date().toLocaleTimeString('en-IN', { hour12: false });
}

function triggerSimulation() {
  if (!state.running) { startSimMode(); return; }
  alertCooldown = false;
  dispatchAlert({
    fighting: true, threat: 0.88, numPersons: 2,
    personResults: [
      { pose: 'PUNCH', threat: 0.88, id: 0 },
      { pose: 'GUARD', threat: 0.65, id: 1 },
    ]
  });
}

function clearAlerts() {
  document.getElementById('chatLog').innerHTML = '';
  state.alertCount = 0;
  document.getElementById('alertBadge').style.display = 'none';
  document.getElementById('sbAlerts').textContent = '0';
  logMsg('Alert log cleared.', 'sys');
}

function dismissBanner() {
  document.getElementById('alertBanner').classList.remove('show');
}

// ========================================================
// CAMERA UI STATES
// ========================================================
function onCameraStarted() {
  document.getElementById('feedPlaceholder').style.display = 'none';
  document.getElementById('feedPills').style.display = '';
  document.getElementById('startBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = '';
  document.getElementById('recDot').className = 'rec-dot live';
  document.getElementById('statusDot').className = 'status-dot live';
  document.getElementById('statusText').textContent = 'Live';
  document.getElementById('pageSubtitle').textContent =
    state.simMode ? 'Simulation mode — 2 person fight cycle' : 'MoveNet MultiPose active — up to 6 persons';
}

function onCameraStopped() {
  document.getElementById('feedPlaceholder').style.display = '';
  document.getElementById('feedPills').style.display = 'none';
  document.getElementById('startBtn').style.display = '';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('recDot').className = 'rec-dot';
  document.getElementById('statusDot').className = 'status-dot';
  document.getElementById('statusText').textContent = 'Offline';
  document.getElementById('pageSubtitle').textContent = 'Camera offline — press Start to begin';
  if (simAnimId) { cancelAnimationFrame(simAnimId); simAnimId = null; }
}

// ========================================================
// SESSION TIMER + TABS + SETTINGS
// ========================================================
setInterval(() => {
  if (!state.sessionStart) return;
  const s  = Math.floor((Date.now() - state.sessionStart) / 1000);
  document.getElementById('sbTime').textContent =
    String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
}, 1000);

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    const tab = item.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('tab-' + tab)?.classList.add('active');
  });
});

function updateSettingLabel(inputId, labelId, suffix, mult) {
  const val = parseFloat(document.getElementById(inputId).value);
  document.getElementById(labelId).textContent = Math.round(val * mult) / mult + suffix;
}

// ========================================================
// LOG + CHAT
// ========================================================
function logMsg(html, type = 'sys') {
  const log = document.getElementById('chatLog');
  const avClass = type === 'alert' ? 'av-alert' : type === 'ok' ? 'av-ok' : 'av-sys';
  const avLabel = type === 'alert' ? '!' : type === 'ok' ? '✓' : 'S';
  const bubClass = type === 'alert' ? 'c-bubble alert-bubble' : 'c-bubble';
  const el = document.createElement('div');
  el.className = 'chat-msg';
  el.innerHTML = `<div class="c-avatar ${avClass}">${avLabel}</div><div class="${bubClass}">${html}<div class="ts">${timestamp()}</div></div>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

// ========================================================
// INIT — update HTML labels for renamed metric cards
// ========================================================
document.addEventListener('DOMContentLoaded', () => {
  // Rename metric card labels to match new data
  const labels = document.querySelectorAll('.mc-label');
  if (labels[1]) labels[1].textContent = 'Wrists elevated';
  if (labels[2]) labels[2].textContent = 'Detected poses';
});

logMsg('Brawl Monitor v3.0 — MoveNet MultiPose (up to 6 persons).', 'sys');
logMsg('Pose classes: Punch · Kick · Guard · Grab · Neutral.', 'sys');
logMsg('Click "Start camera" or "Simulate fight" to begin.', 'sys');
