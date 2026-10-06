/**
 * FlyLab front end.
 *
 * Two live views share one WebSocket:
 *   - a WebGL point cloud of all 138,639 neurons, where each point brightens
 *     with that neuron's firing rate (the "glowing brain")
 *   - the MuJoCo camera view of the fly, streamed as JPEG frames
 *
 * Activity arrives as one uint8 per neuron per tick, uploaded straight into a
 * GPU attribute. Doing this in JSON would be ~40x the bytes and would stall the
 * main thread, so the socket speaks a small binary protocol instead.
 */

import * as THREE from "/vendor/three.module.js";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { EffectComposer } from "/vendor/EffectComposer.js";
import { RenderPass } from "/vendor/RenderPass.js";
import { UnrealBloomPass } from "/vendor/UnrealBloomPass.js";
import { OutputPass } from "/vendor/OutputPass.js";

const REGION_COLORS = {
  optic_lobe: 0x2f7bff,
  visual_projection: 0x00d0ff,
  sensory: 0x36e39b,
  descending: 0xff5fd2,
  ascending: 0xffb545,
  motor: 0xff4d5e,
  endocrine: 0xc084fc,
  central: 0x8fa2c9,
};

const NT_COLORS = {
  acetylcholine: 0x59f2b0,
  glutamate: 0x4da6ff,
  gaba: 0xff6b8a,
  dopamine: 0xffc542,
  serotonin: 0xc084fc,
  octopamine: 0x00e5ff,
  unknown: 0x6b7690,
};

const REGION_LABELS = {
  optic_lobe: ["Optic lobes", "vision, motion detection"],
  visual_projection: ["Visual projection", "eyes to central brain"],
  sensory: ["Sensory afferents", "taste, touch, smell, wind"],
  descending: ["Descending neurons", "brain to legs and wings"],
  ascending: ["Ascending neurons", "body back to brain"],
  motor: ["Motor neurons", "muscle output"],
  endocrine: ["Endocrine", "hormonal state"],
  central: ["Central brain", "memory, navigation, decisions"],
};

const state = {
  meta: null,
  n: 0,
  activity: null,
  regionCodes: null,
  ntCodes: null,
  colorMode: "region",
  lesioned: new Set(),
  paused: false,
  spikeFlash: true,
  lastFrameUrl: null,
};

const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ 3D brain */
const canvas = el("brain-canvas");
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x03050a, 0.0016);

const camera = new THREE.PerspectiveCamera(45, 1, 1, 6000);
camera.position.set(0, -120, 620);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.rotateSpeed = 0.6;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;

/**
 * Bloom post-processing.
 *
 * The glow around a firing neuron is light spilling from that neuron, not a
 * bigger sprite. UnrealBloomPass extracts pixels brighter than `threshold`,
 * blurs them across several mip levels, and adds the result back. That is what
 * produces the heat-map-like bloom while every neuron keeps its own size,
 * colour, and blink.
 */
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(1, 1), // real size is set in resize()
  0.95, // strength: how bright the spilled light is
  0.30, // radius: kept small so light stays near the neuron, not on the background
  0.40, // threshold: only clearly active cells emit, so idle structure stays crisp
);
const outputPass = new OutputPass();
composer.addPass(renderPass);
composer.addPass(bloomPass);
composer.addPass(outputPass);

/**
 * Point shader. Base colour comes from a per-neuron attribute; `aAct` is the
 * live firing rate. Active neurons grow slightly and gain an additive white-hot
 * core, which reads as a glow once many of them light up together.
 */
const material = new THREE.ShaderMaterial({
  uniforms: {
    uSize: { value: 1.9 },
    uScale: { value: 620.0 },
    uDim: { value: 0.16 },
    uHeat: { value: 0.0 },
  },
  vertexShader: `
    attribute vec3 aColor;
    attribute float aAct;
    attribute float aMuted;
    uniform float uSize;
    uniform float uScale;
    uniform float uDim;
    uniform float uHeat;
    varying vec3 vColor;
    varying float vAct;
    void main() {
      vAct = aAct;
      vec3 base = mix(aColor, vec3(1.0, 0.55, 0.15), uHeat);
      // A firing neuron keeps its region hue and is pushed above 1.0 so the
      // bloom pass, which only picks up pixels over its threshold, treats it as
      // a light source. The surrounding glow is produced by that pass, not by
      // inflating this sprite.
      float lit = clamp(aAct * 1.9, 0.0, 1.0);
      vec3 hot = base * (1.0 + lit * 1.15);
      vColor = mix(base * uDim, hot, lit);
      vColor *= mix(1.0, 0.25, aMuted);
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      // Firing cells grow only slightly. Bloom spreads the light outward, so
      // the sprite stays close to the neuron's actual footprint.
      float size = uSize * (0.85 + lit * 0.9);
      gl_PointSize = size * (uScale / -mv.z);
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    varying float vAct;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      float r = length(d);
      if (r > 0.5) discard;
      float lit = clamp(vAct * 1.9, 0.0, 1.0);
      // Compact round sprite: a soft edge for idle points and a tighter, more
      // solid disc when firing, so the bloom pass has a crisp emitter to blur.
      float disc = smoothstep(0.5, 0.12, r);
      float alpha = clamp(disc * (0.16 + lit * 0.62), 0.0, 1.0);
      gl_FragColor = vec4(vColor, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});

let points = null;
let actAttr = null;
let colorAttr = null;
let mutedAttr = null;
// Camera distance at the default framing, used as the glow reference point.
let glowReferenceDistance = 0;
// Bounding radius of the neuron cloud, in micrometres.
let brainRadius = 0;
// Render diagnostics overlay, toggled with the `d` key.
let debugVisible = false;

function resize() {
  const view = el("brain-view");
  const w = view.clientWidth;
  const h = view.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  material.uniforms.uScale.value = h * 0.85;
  composer.setSize(w, h);
  bloomPass.resolution.set(w, h);
}
window.addEventListener("resize", resize);

async function loadBrainGeometry() {
  const res = await fetch("/api/points");
  if (!res.ok) throw new Error("point buffer unavailable");
  const raw = new Float32Array(await res.arrayBuffer());
  const n = raw.length / 5;
  state.n = n;

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const act = new Float32Array(n);
  const muted = new Float32Array(n);
  const regionCodes = new Uint8Array(n);
  const ntCodes = new Uint8Array(n);

  for (let i = 0; i < n; i += 1) {
    const o = i * 5;
    pos[i * 3] = raw[o];
    pos[i * 3 + 1] = raw[o + 1];
    pos[i * 3 + 2] = raw[o + 2];
    regionCodes[i] = raw[o + 3];
    ntCodes[i] = raw[o + 4];
  }
  state.regionCodes = regionCodes;
  state.ntCodes = ntCodes;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  colorAttr = new THREE.BufferAttribute(col, 3);
  actAttr = new THREE.BufferAttribute(act, 1);
  mutedAttr = new THREE.BufferAttribute(muted, 1);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  actAttr.setUsage(THREE.DynamicDrawUsage);
  mutedAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute("aColor", colorAttr);
  geom.setAttribute("aAct", actAttr);
  geom.setAttribute("aMuted", mutedAttr);
  geom.computeBoundingSphere();

  points = new THREE.Points(geom, material);
  scene.add(points);

  // Frame the brain: FlyWire coordinates put the fly nose-down, so tilt it up.
  points.rotation.x = -Math.PI / 2;
  const r = geom.boundingSphere ? geom.boundingSphere.radius : 400;
  brainRadius = r;
  glowReferenceDistance = r * 1.62;
  // Keep the cloud reachable but not escapable: without limits, a few scroll
  // steps put the camera inside the brain or far past it into empty space.
  // The near limit also bounds how large the glow can grow on screen, because
  // the sprite keeps a minimum size that no longer shrinks with distance.
  // 0.6x keeps the camera outside the cloud: closer than that and it sits
  // inside the shell, looking through mostly empty interior.
  controls.minDistance = r * 0.6;
  controls.maxDistance = r * 3.0;
  resetView();
  applyColorMode();
  state.activity = new Uint8Array(n);
  rateTarget = new Uint8Array(n);
  resize();
}

/**
 * Return the camera to the default framing.
 *
 * Orbiting can leave the target drifted off the cloud, so this restores both the
 * position and the target rather than only the distance.
 */
function resetView() {
  if (!brainRadius) return;
  camera.position.set(0, -brainRadius * 0.28, brainRadius * 1.62);
  controls.target.set(0, 0, 0);
  controls.update();
}

function applyColorMode() {
  if (!colorAttr || !state.meta) return;
  const { regions, neurotransmitters } = state.meta;
  const arr = colorAttr.array;
  const c = new THREE.Color();
  for (let i = 0; i < state.n; i += 1) {
    let hex;
    if (state.colorMode === "nt") {
      hex = NT_COLORS[neurotransmitters[state.ntCodes[i]]] ?? 0x6b7690;
    } else if (state.colorMode === "heat") {
      hex = 0x3b4a68;
    } else {
      hex = REGION_COLORS[regions[state.regionCodes[i]]] ?? 0x8fa2c9;
    }
    c.setHex(hex);
    arr[i * 3] = c.r;
    arr[i * 3 + 1] = c.g;
    arr[i * 3 + 2] = c.b;
  }
  colorAttr.needsUpdate = true;
  material.uniforms.uHeat.value = state.colorMode === "heat" ? 0.85 : 0.0;
}

function updateLesionShading() {
  if (!mutedAttr || !state.meta) return;
  const arr = mutedAttr.array;
  const codes = state.regionCodes;
  const regions = state.meta.regions;
  for (let i = 0; i < state.n; i += 1) {
    arr[i] = state.lesioned.has(regions[codes[i]]) ? 1 : 0;
  }
  mutedAttr.needsUpdate = true;
}

/**
 * Latest sustained firing rate per neuron, straight from the simulation.
 * Kept separate from the rendered value so the afterglow can decay on its own
 * clock instead of being overwritten by every incoming tick.
 */
let rateTarget = null;

/** Push the latest per-neuron rates in as the floor the afterglow decays to. */
function uploadActivity(bytes) {
  if (!rateTarget || bytes.length !== state.n) return;
  rateTarget.set(bytes);
}

/** Latch neurons that just spiked to full brightness. */
function flashSpikes(indices) {
  if (!state.spikeFlash || !actAttr) return;
  const arr = actAttr.array;
  for (let k = 0; k < indices.length; k += 1) {
    const i = indices[k];
    if (i >= 0 && i < arr.length) arr[i] = 1.0;
  }
}

/**
 * Fade every point toward its sustained rate.
 *
 * Runs per animation frame rather than per simulation tick, so a spike leaves a
 * visible trail that dims smoothly instead of vanishing on the next update. The
 * decay is exponential and frame-rate corrected, so the trail lasts the same
 * wall-clock time whether the browser runs at 30 or 120 fps.
 */
const GLOW_HALF_LIFE_MS = 260;

function decayActivity(dtMs) {
  if (!actAttr || !rateTarget) return;
  const arr = actAttr.array;
  const keep = Math.pow(0.5, dtMs / GLOW_HALF_LIFE_MS);
  for (let i = 0; i < arr.length; i += 1) {
    const floor = rateTarget[i] * (1 / 255);
    const v = arr[i];
    // Decay toward the sustained rate, never below it.
    arr[i] = v > floor ? floor + (v - floor) * keep : floor;
  }
  actAttr.needsUpdate = true;
}

/**
 * Scale the halo down as the camera closes in.
 *
 * Sprite size is perspective-divided, so without this the halo grows on screen
 * exactly as fast as the zoom and inspecting a small cluster means staring into
 * a wall of bloom. Tying the multiplier to camera distance keeps the glow
 * roughly constant in screen space.
 */
function updateGlowScale() {
  if (!points || !glowReferenceDistance) return;
  const dist = camera.position.distanceTo(controls.target);
  // Bloom radius is measured in screen space, so zooming in makes each
  // neuron's light spill across proportionally more of the view. Easing the
  // radius and strength down as the camera closes in keeps a close-up readable
  // instead of drowning it in the neighbours' light.
  const ratio = Math.min(1.0, Math.max(0.0, dist / glowReferenceDistance));
  bloomPass.radius = 0.14 + 0.16 * ratio;
  bloomPass.strength = 0.47 + 0.48 * ratio;
  if (debugVisible) {
    const t = controls.target;
    el("brain-debug").textContent =
      `dist ${dist.toFixed(0)} / ref ${glowReferenceDistance.toFixed(0)}\n` +
      `bloom str ${bloomPass.strength.toFixed(2)} rad ${bloomPass.radius.toFixed(2)}\n` +
      `limits ${controls.minDistance.toFixed(0)}-${controls.maxDistance.toFixed(0)}\n` +
      `target ${t.x.toFixed(0)},${t.y.toFixed(0)},${t.z.toFixed(0)}`;
  }
}

let lastFrameMs = performance.now();

function animate() {
  const now = performance.now();
  // Clamp so a backgrounded tab does not wipe the trail in one huge step.
  const dtMs = Math.min(120, now - lastFrameMs);
  lastFrameMs = now;

  controls.update();
  updateGlowScale();
  decayActivity(dtMs);
  composer.render();
  requestAnimationFrame(animate);
}

/* ------------------------------------------------------- click to inspect */
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 6;
const pointer = new THREE.Vector2();

canvas.addEventListener("click", async (ev) => {
  if (!points) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(points, false);
  if (!hits.length) {
    el("inspect").hidden = true;
    return;
  }
  const idx = hits[0].index;
  const res = await fetch(`/api/neuron/${idx}`);
  if (!res.ok) return;
  const d = await res.json();
  const box = el("inspect");
  box.hidden = false;
  box.innerHTML = `
    <h3>${d.cell_type || "unnamed cell"}</h3>
    <div><span>region</span><span>${d.region}</span></div>
    <div><span>transmitter</span><span>${d.neurotransmitter}</span></div>
    <div><span>side</span><span>${d.side}</span></div>
    <div><span>outputs</span><span>${d.out_degree.toLocaleString()}</span></div>
    <div><span>rate</span><span>${d.rate_hz.toFixed(1)} Hz</span></div>
    <div><span>membrane</span><span>${d.membrane_mv.toFixed(1)} mV</span></div>
    <div><span>FlyWire id</span><span>${d.root_id.slice(0, 10)}…</span></div>
    <button data-stim="${d.root_id}">stimulate this neuron</button>
  `;
  box.querySelector("button").addEventListener("click", async () => {
    await post("/api/stimulate", { root_ids: [Number(d.root_id)], rate_hz: 200 });
  });
});

/* --------------------------------------------------------------- transport */
async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return res.ok ? res.json() : null;
}

function decodeFrame(buf) {
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
  );
  let off = 4 + headerLen;
  const activity = new Uint8Array(buf, off, header.n_activity);
  off += header.n_activity;
  const spikes = header.n_spikes
    ? new Int32Array(buf.slice(off, off + header.n_spikes * 4))
    : null;
  off += header.n_spikes * 4;
  const jpeg = header.n_jpeg ? new Uint8Array(buf, off, header.n_jpeg) : null;
  return { header, activity, spikes, jpeg };
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => el("link-dot").classList.add("live"));
  ws.addEventListener("close", () => {
    el("link-dot").classList.remove("live");
    setTimeout(connect, 1200);
  });
  ws.addEventListener("message", (ev) => {
    const { header, activity, spikes, jpeg } = decodeFrame(ev.data);
    uploadActivity(activity);
    if (spikes) flashSpikes(spikes);
    renderStats(header);
    renderRegions(header.region_rates);
    renderBody(header.body);
    if (jpeg) showFrame(jpeg);
  });
}

function showFrame(bytes) {
  const blob = new Blob([bytes], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  const img = el("fly-frame");
  img.src = url;
  if (state.lastFrameUrl) URL.revokeObjectURL(state.lastFrameUrl);
  state.lastFrameUrl = url;
}

/* ------------------------------------------------------------------- panels */
let spikeRateEma = 0;

function renderStats(h) {
  const ms = h.brain_time_ms;
  el("stat-time").textContent =
    ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`;
  el("stat-rt").textContent = `${h.realtime_factor.toFixed(2)}\u00d7`;
  const perSec = h.spike_count * h.wall_hz;
  spikeRateEma = spikeRateEma * 0.8 + perSec * 0.2;
  el("stat-spikes").textContent = Math.round(spikeRateEma).toLocaleString();
  el("stat-hz").textContent = `${h.wall_hz.toFixed(0)} Hz`;
}

function renderRegions(rates) {
  const list = el("regions");
  if (!list.children.length) {
    for (const name of state.meta.regions) {
      const [title, sub] = REGION_LABELS[name] ?? [name, ""];
      const li = document.createElement("li");
      li.className = "region";
      li.dataset.region = name;
      li.title = "Click to lesion this region";
      const color = `#${(REGION_COLORS[name] ?? 0x8fa2c9)
        .toString(16)
        .padStart(6, "0")}`;
      li.innerHTML = `
        <span class="swatch" style="background:${color}"></span>
        <span class="name"><b>${title}</b><small>${sub} · ${(
          state.meta.region_counts[name] ?? 0
        ).toLocaleString()} neurons</small></span>
        <span class="bar"><i style="background:${color}"></i></span>
      `;
      li.addEventListener("click", () => toggleLesion(name, li));
      list.appendChild(li);
    }
  }
  for (const li of list.children) {
    const name = li.dataset.region;
    const hz = rates[name] ?? 0;
    const bar = li.querySelector(".bar i");
    const pct = Math.min(100, (hz / 40) * 100);
    bar.style.width = `${pct}%`;
    const swatch = li.querySelector(".swatch");
    const glow = Math.min(1, hz / 25);
    swatch.style.boxShadow = glow > 0.02
      ? `0 0 ${4 + glow * 12}px rgba(255,255,255,${0.25 + glow * 0.6})`
      : "none";
  }
}

function renderBody(body) {
  if (!body || !Object.keys(body).length) return;
  el("behavior-badge").textContent = body.behavior ?? "—";
  const legs = el("legs");
  if (!legs.children.length) {
    for (let i = 0; i < 6; i += 1) {
      const d = document.createElement("div");
      d.className = "leg";
      d.title = ["left front", "left mid", "left hind",
                 "right front", "right mid", "right hind"][i];
      legs.appendChild(d);
    }
  }
  const order = [0, 3, 1, 4, 2, 5];
  (body.contacts ?? []).forEach((_, i) => {
    const cell = legs.children[order.indexOf(i)];
    if (cell) cell.classList.toggle("down", Boolean(body.contacts[i]));
  });

  const rows = [
    ["speed", `${(body.speed_mm_s ?? 0).toFixed(1)} mm/s`],
    ["walked", `${(body.distance_mm ?? 0).toFixed(1)} mm`],
    ["heading", `${(body.heading_deg ?? 0).toFixed(0)}\u00b0`],
    ["descending drive", (body.drive ?? 0).toFixed(2)],
    ["turn bias", (body.turn_bias ?? 0).toFixed(2)],
    ["body clock", `${(body.sim_time_s ?? 0).toFixed(2)} s`],
  ];
  el("body-readout").innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");
}

async function toggleLesion(name, li) {
  const on = !state.lesioned.has(name);
  await post("/api/silence", { region: name, silenced: on });
  if (on) state.lesioned.add(name);
  else state.lesioned.delete(name);
  li.classList.toggle("lesioned", on);
  updateLesionShading();
}

/* ------------------------------------------------------------------ controls */
function buildPresets() {
  const box = el("presets");
  for (const [key, info] of Object.entries(state.meta.presets)) {
    const b = document.createElement("button");
    b.className = "preset" + (key === "rest" ? " active" : "");
    b.textContent = info.label;
    b.title = info.description;
    b.addEventListener("click", async () => {
      await post("/api/preset", { name: key });
      for (const other of box.children) other.classList.remove("active");
      b.classList.add("active");
    });
    box.appendChild(b);
  }
}

function wireControls() {
  el("opt-spin").addEventListener("change", (e) => {
    controls.autoRotate = e.target.checked;
  });
  el("opt-trails").addEventListener("change", (e) => {
    state.spikeFlash = e.target.checked;
  });
  el("opt-color").addEventListener("change", (e) => {
    state.colorMode = e.target.value;
    applyColorMode();
  });
  const speed = el("speed");
  speed.addEventListener("input", () => {
    el("speed-out").textContent = `${Number(speed.value).toFixed(1)}\u00d7`;
  });
  speed.addEventListener("change", () => {
    post("/api/speed", { scale: Number(speed.value) });
  });
  el("btn-pause").addEventListener("click", async (e) => {
    state.paused = !state.paused;
    await post("/api/pause", { paused: state.paused });
    e.target.textContent = state.paused ? "resume" : "pause";
    e.target.classList.toggle("on", state.paused);
  });
  el("btn-reset").addEventListener("click", () => post("/api/reset"));
  // `d` toggles the render diagnostics overlay, and `r` reframes the brain.
  window.addEventListener("keydown", (ev) => {
    if (ev.target instanceof HTMLInputElement) return;
    if (ev.key === "d") {
      debugVisible = !debugVisible;
      el("brain-debug").hidden = !debugVisible;
    } else if (ev.key === "r") {
      resetView();
    }
  });
  el("btn-unsilence").addEventListener("click", async () => {
    await post("/api/silence/clear");
    state.lesioned.clear();
    for (const li of el("regions").children) li.classList.remove("lesioned");
    updateLesionShading();
  });
}

/* ---------------------------------------------------------------- bootstrap */
async function main() {
  state.meta = await (await fetch("/api/meta")).json();
  el("scale-line").textContent =
    `${state.meta.n_neurons.toLocaleString()} neurons · ` +
    `${(state.meta.n_edges / 1e6).toFixed(1)}M synaptic connections`;
  el("source").textContent =
    `${state.meta.source.connectome} · ${state.meta.source.model}`;
  if (!state.meta.has_body) {
    el("behavior-badge").textContent = "body offline";
  }
  buildPresets();
  wireControls();
  await loadBrainGeometry();
  animate();
  connect();
}

main().catch((err) => {
  el("scale-line").textContent = `failed to start: ${err.message}`;
  console.error(err);
});
