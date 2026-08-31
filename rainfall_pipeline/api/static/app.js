/* Forecasting Lab — rainfall post-processing console.
 *
 * Everything drawn here comes from the API at request time. There are no
 * hard-coded rainfall numbers, no baked-in accuracy scores and no placeholder
 * series: if the service has no data, the page says so rather than inventing
 * something that looks like a result.
 *
 * Two rules govern this file:
 *   1. The pipeline's own vocabulary (column names, verification acronyms)
 *      never reaches the screen unless the reader has asked for it. All
 *      translations live in the vocabulary block below.
 *   2. Resolution is never implied beyond what the data carries. The dataset is
 *      daily, so the trend chart plots days. There is no hourly interpolation
 *      anywhere, because there is no hourly data.
 */
"use strict";

const REGIME_ORDER = ["Active", "Break", "Depression-Low", "Coastal", "Orographic"];

/* Sequential rainfall ramp for the gridded panels, tuned for a slate ground:
 * deep navy through teal and green into the warm colours for extremes. */
const RAIN_RAMP = [
  [ 13,  21,  38], [ 20,  48,  86], [ 24,  86, 140], [ 32, 132, 178],
  [ 45, 178, 176], [ 60, 190, 130], [128, 205,  86], [216, 208,  74],
  [244, 166,  52], [232, 100,  52], [196,  44,  56], [150,  36, 112],
];

/* Diverging ramp for the correction panel: rose = the model took rainfall
 * away, sky = it added rainfall, dark = it left the forecast alone. */
const DIFF_RAMP = [
  [158,  32,  50], [206,  70,  74], [190, 110, 116], [ 84,  62,  84],
  [ 19,  27,  46], [ 46,  86, 128], [ 56, 132, 190], [ 46, 118, 220],
  [ 38,  96, 232],
];

/* IMD daily rainfall category boundaries (mm). Rainfall is far from uniformly
 * distributed -- most cells on most days are light -- so a linear colour scale
 * washes the whole map out and hides exactly the structure a forecaster is
 * looking for. Colour is assigned by band instead. */
const RAIN_BANDS = [0, 2.5, 15.6, 64.5, 115.6, 204.4];

/* Light-ground variants. On white, a dark-navy low end reads as "heavy rain
 * here" instead of "nothing here", so the dry end has to be the palest value
 * rather than the deepest. */
const RAIN_RAMP_LIGHT = [
  [246, 249, 253], [219, 234, 247], [180, 214, 238], [124, 187, 222],
  [ 74, 172, 175], [ 74, 176, 122], [136, 199,  86], [226, 205,  74],
  [240, 158,  48], [223,  95,  46], [186,  38,  44], [140,  34, 110],
];

const DIFF_RAMP_LIGHT = [
  [158,  32,  50], [206,  84,  88], [230, 150, 156], [242, 208, 212],
  [248, 250, 253], [204, 224, 242], [148, 192, 228], [ 74, 142, 206],
  [ 24,  96, 184],
];

/* True when the console is showing its light ground. */
function isLight() {
  return document.documentElement.getAttribute("data-theme") === "light";
}

function rainRamp() { return isLight() ? RAIN_RAMP_LIGHT : RAIN_RAMP; }
function diffRamp() { return isLight() ? DIFF_RAMP_LIGHT : DIFF_RAMP; }

/* Canvas drawing cannot use CSS variables directly, so read them off the root
 * element at draw time. This keeps one source of truth for every colour. */
function themeColour(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (value && value.trim()) || fallback;
}

/* ------------------------------------------------------------- vocabulary
 *
 * The pipeline speaks in column names and verification acronyms. Nobody
 * outside the field reads `cape_humidity` or knows what FAR stands for, and a
 * console that shows those is only legible to the person who built it.
 * Everything user-facing is translated here; the raw names stay available in
 * the technical scorecard for the people who do want them.
 */

/* Feature column -> what it physically means. */
const FEATURE_LABELS = {
  raw_forecast_mm: "The weather model's own forecast",
  log_raw_forecast: "The weather model's forecast (fine detail)",
  observed_mm: "Measured rainfall",
  cape: "How unstable the air is",
  cape_humidity: "Unstable air combined with moisture",
  humidity: "How much moisture is in the air",
  pressure_msl: "Air pressure at sea level",
  pressure_anomaly: "How unusual the air pressure is",
  vorticity: "How much the air is spinning",
  olr: "How cold and high the cloud tops are",
  wind_u_850: "Low-level wind, east–west",
  wind_v_850: "Low-level wind, north–south",
  wind_u_200: "High-level wind, east–west",
  wind_v_200: "High-level wind, north–south",
  wind_speed_850: "Low-level wind strength",
  wind_speed_200: "High-level wind strength",
  wind_dir_850: "Low-level wind direction",
  wind_shear: "How much the wind changes with height",
  shear_u: "Wind change with height, east–west",
  shear_v: "Wind change with height, north–south",
  moisture_flux_850: "Moisture being carried into the area",
  onshore_flow: "Wind blowing in from the sea",
  upslope_flow: "Wind being pushed up a slope",
  elevation: "How high the land is",
  coastal_distance: "How far this is from the sea",
  is_coastal: "Whether this is near the coast",
  is_orographic: "Whether this is hilly ground",
  rain_anomaly_sd: "How unusual the rainfall is for this place",
  lat: "Where it is, north–south",
  lon: "Where it is, east–west",
  day_of_year: "Where we are in the season",
  doy_sin: "Where we are in the season",
  doy_cos: "Where we are in the season",
  month: "Which month it is",
  lead_time: "How far ahead the forecast is",
  ivt_proxy: "Integrated moisture transport in the lower atmosphere",
  convective_instability: "Deep convective instability index",
  forecast_spatial_mean_3x3: "Nearby surrounding average forecast rain",
  forecast_spatial_max_3x3: "Nearby peak forecast rain intensity",
  forecast_spatial_std_3x3: "Nearby forecast rainfall variability",
  cape_spatial_max_3x3: "Nearby maximum atmospheric instability",
  moisture_flux_spatial_mean_3x3: "Nearby surrounding average moisture flux",
  upwind_forecast_rain: "Forecast rainfall in the upwind direction",
  upwind_moisture_flux: "Moisture flowing in from the upwind direction",
};

/* Weather type -> a one-line description of the physical situation. */
const REGIME_INFO = {
  Active: {
    short: "Active monsoon",
    description: "The monsoon is in full swing — widespread, steady rain.",
    colour: "#22c55e",
  },
  Break: {
    short: "Monsoon break",
    description: "A lull in the monsoon — most places stay dry or nearly so.",
    colour: "#c08a2e",
  },
  "Depression-Low": {
    short: "Low-pressure system",
    description: "A storm system is sitting overhead — heavy, concentrated rain.",
    colour: "#f43f5e",
  },
  Coastal: {
    short: "Coastal rain",
    description: "Sea winds are piling into the shoreline and dumping rain there.",
    colour: "#a78bfa",
  },
  Orographic: {
    short: "Hill rain",
    description: "Wind is being forced up over high ground, wringing rain out of it.",
    colour: "#38bdf8",
  },
};

/* Model id -> plain name, plus why it is in the comparison at all. */
const MODEL_INFO = {
  A_raw_nwp: {
    name: "The weather model, uncorrected",
    role: "What forecasters use today. Anything we build has to beat this.",
  },
  B_global_ml: {
    name: "AI correction, one model for everything",
    role: "Same AI, but blind to what kind of weather it is. This is the one our system must beat to prove that weather type matters.",
  },
  C_quantile_mapping: {
    name: "Statistical correction, no AI",
    role: "The classic textbook fix. Shows how much of the gain needs AI at all.",
  },
  C_regime_quantile_mapping: {
    name: "Statistical correction, per weather type",
    role: "Weather-type-aware but with no AI. Separates 'knowing the weather type helps' from 'AI helps'.",
  },
  D_regime_residual: {
    name: "Our system",
    role: "AI correction with a separate specialist for each kind of weather.",
  },
  E_regime_residual_probability: {
    name: "Our system, plus risk percentages",
    role: "Same rainfall as our system, with calibrated odds of dangerous rain added on top.",
  },
};

const OUR_MODEL = "D_regime_residual";
const BLIND_MODEL = "B_global_ml";
const RAW_MODEL = "A_raw_nwp";

const THRESHOLD_LABEL = {
  heavy: "Heavy rain",
  very_heavy: "Very heavy rain",
  extremely_heavy: "Extreme rain",
};

const THRESHOLD_MM = { heavy: 64.5, very_heavy: 115.6, extremely_heavy: 204.4 };

const WARNING_TEXT = {
  none: {
    headline: "NO ADVISORY",
    detail: "Rainfall is expected to stay below the heavy-rain threshold.",
  },
  watch: {
    headline: "KEEP WATCH",
    detail: "There is a realistic chance of heavy rain. Worth monitoring.",
  },
  warning: {
    headline: "WARNING",
    detail: "Heavy rain is more likely than not. Prepare for disruption.",
  },
  severe: {
    headline: "SEVERE WARNING",
    detail: "Very heavy rain is likely. Treat this as a serious risk.",
  },
};

function featureLabel(name) {
  return FEATURE_LABELS[name] || name.replace(/_/g, " ");
}

function regimeInfo(name) {
  return REGIME_INFO[name] || { short: name, description: "", colour: "#64748b" };
}

function modelInfo(id) {
  return MODEL_INFO[id] || { name: id, role: "" };
}

/* Split a compound label like "Coastal + Orographic" into plain names. */
function plainRegimeLabel(label) {
  if (!label) return "";
  return String(label).split(" + ").map((r) => regimeInfo(r).short).join(" + ");
}

const el = (id) => document.getElementById(id);
const state = {
  dates: null, health: null, districtSource: null,
  lastGrid: null, lastPrediction: null, lastReport: null, lastTimeline: null,
  staticMode: false, demo: undefined,
  view: "overview",
};

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const n = Number(value);
  // Rounding a real 0.4% risk down to "0%" reads as "cannot happen", which is
  // a different claim from "unlikely".
  if (n > 0 && n < 0.005) return "<1%";
  if (n > 0.995 && n < 1) return ">99%";
  return `${Math.round(n * 100)}%`;
}

/* Round a rainfall amount the way a person would say it out loud. */
function mm(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const n = Number(value);
  return `${n >= 100 ? Math.round(n) : n.toFixed(1)} mm`;
}

/* Bare number for the metric cards, where the unit is its own element. */
function num(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const n = Number(value);
  return n >= 100 ? String(Math.round(n)) : n.toFixed(1);
}

function prettyDate(iso) {
  const parsed = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
}

function shortDate(iso) {
  const parsed = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function showAlert(message, kind = "error") {
  const node = el("alert");
  node.textContent = message;
  node.classList.remove("hidden", "info");
  if (kind === "info") node.classList.add("info");
}

function clearAlert() { el("alert").classList.add("hidden"); }

/* ------------------------------------------------------------ static mode
 *
 * The dashboard normally reads a live service at the same origin. On a static
 * host there is none, so the first failed call flips the page into static
 * mode: every later request is answered from the frozen snapshot in demo/.
 * The snapshot is one date only, and the page says so rather than letting the
 * date picker imply a range it cannot serve.
 */

const DEMO_DIR = "demo";

function slugify(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "unnamed";
}

/* Map a live request onto its snapshot file. Returns null when the endpoint
 * cannot meaningfully be frozen -- the what-if probe re-runs the model per
 * request, so a stale answer would be a lie rather than a fallback. */
function demoFileFor(path) {
  const manifest = state.demo;
  if (!manifest) return null;
  const [route, query = ""] = path.split("?");
  const params = new URLSearchParams(query);
  const files = manifest.files || {};
  const district = params.get("district");

  switch (route) {
    case "/health": return files.health;
    case "/dates": return files.dates;
    case "/districts": return files.districts;
    case "/verification-report": return files["verification-report"];
    case "/grid": return files.grid;
    case "/watch": return files.watch;
    case "/risk-matrix": return files["risk-matrix"];
    case "/drivers": return files.drivers;
    case "/events":
      return files[`events:${params.get("unseen_only") === "false" ? "false" : "true"}`];
    case "/predict":
      return district ? files[`predict:${slugify(district)}`] : null;
    case "/timeline":
      return district ? files[`timeline:${slugify(district)}`] : null;
    default:
      return null;
  }
}

async function loadDemoManifest() {
  if (state.demo !== undefined) return state.demo;
  try {
    const response = await fetch(`${DEMO_DIR}/index.json`, { cache: "no-store" });
    state.demo = response.ok ? await response.json() : null;
  } catch (err) {
    state.demo = null;
  }
  return state.demo;
}

async function getJSON(path) {
  if (state.staticMode) {
    const file = demoFileFor(path);
    if (!file) {
      throw new Error(
        "This panel needs the live service. The hosted demo is a frozen " +
        "snapshot, so it cannot re-run the model for a new request."
      );
    }
    const snap = await fetch(`${DEMO_DIR}/${file}`, { cache: "no-store" });
    if (!snap.ok) throw new Error(`Snapshot ${file} is missing from this deployment.`);
    return snap.json();
  }

  let response;
  try {
    response = await fetch(path);
  } catch (err) {
    // No service at this origin at all -- try the frozen snapshot once.
    const manifest = await loadDemoManifest();
    if (manifest) {
      state.staticMode = true;
      enterStaticMode(manifest);
      return getJSON(path);
    }
    throw new Error(`Could not reach ${path}: ${err.message}`);
  }
  if (response.status === 404 && !state.staticMode) {
    const manifest = await loadDemoManifest();
    if (manifest) {
      state.staticMode = true;
      enterStaticMode(manifest);
      return getJSON(path);
    }
  }
  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    throw new Error(`${path} returned a non-JSON response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    const steps = body && Array.isArray(body.next_steps) && body.next_steps.length
      ? `\n\nNext steps:\n  ${body.next_steps.join("\n  ")}`
      : "";
    throw new Error(`${detail}${steps}`);
  }
  return body;
}

function sampleRamp(ramp, t) {
  if (!Number.isFinite(t)) return null;
  const clamped = Math.min(Math.max(t, 0), 1);
  const scaled = clamped * (ramp.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, ramp.length - 1);
  const f = scaled - lo;
  return [0, 1, 2].map((i) => Math.round(ramp[lo][i] + (ramp[hi][i] - ramp[lo][i]) * f));
}

const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

/* Map a rainfall amount onto [0, 1] by IMD band, interpolating within the band
 * so the ramp stays continuous rather than posterised. */
function rainScale(value, top) {
  if (!Number.isFinite(value)) return 0;
  const bands = RAIN_BANDS.filter((b) => b < top).concat([Math.max(top, RAIN_BANDS[1])]);
  const slice = 1 / (bands.length - 1);
  for (let i = 0; i < bands.length - 1; i += 1) {
    if (value < bands[i + 1]) {
      const within = (value - bands[i]) / (bands[i + 1] - bands[i] || 1);
      return (i + Math.min(Math.max(within, 0), 1)) * slice;
    }
  }
  return 1;
}

/* Signed square root, so a 5 mm correction stays visible next to a 90 mm one. */
function diffScale(value, peak) {
  if (!Number.isFinite(value) || peak <= 0) return 0.5;
  const norm = Math.sign(value) * Math.sqrt(Math.abs(value) / peak);
  return 0.5 + Math.min(Math.max(norm, -1), 1) / 2;
}

function hexToRGBA(colour, alpha) {
  const probe = document.createElement("canvas").getContext("2d");
  probe.fillStyle = "#000";
  probe.fillStyle = colour;
  const v = probe.fillStyle;
  if (v.startsWith("#")) {
    const h = v.slice(1);
    const n = h.length === 3
      ? [h[0] + h[0], h[1] + h[1], h[2] + h[2]]
      : [h.slice(0, 2), h.slice(2, 4), h.slice(4, 6)];
    const [r, g, b] = n.map((x) => parseInt(x, 16));
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  const m = v.match(/[\d.]+/g).map(Number);
  return `rgba(${m[0]}, ${m[1]}, ${m[2]}, ${alpha})`;
}

function catmullRom(ctx, points) {
  if (points.length < 2) return;
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || p2;
    ctx.bezierCurveTo(
      p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
      p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6,
      p2.x, p2.y
    );
  }
}


/* Lock the page to what the snapshot can actually answer, and say why. */
function enterStaticMode(manifest) {
  const date = el("date");
  if (date) {
    date.value = manifest.date;
    date.min = manifest.date;
    date.max = manifest.date;
    date.readOnly = true;
    date.title = "The hosted demo is a snapshot of a single date.";
  }
  const hint = el("date-hint");
  if (hint) hint.textContent = `Snapshot of ${manifest.date} — one date only`;

  // A frozen snapshot of fabricated data is still fabricated. Force the notice
  // on and make it permanent.
  const banner = el("synthetic-banner");
  if (banner && manifest.synthetic) {
    banner.classList.remove("hidden");
    const tag = banner.querySelector(".banner-tag");
    if (tag) tag.textContent = "Static demo";
    const text = banner.querySelector("span:last-child");
    if (text) {
      text.textContent =
        "Hosted snapshot of a fabricated demo dataset, frozen at " + manifest.date +
        ". Every value here is made up and says nothing about real rainfall " +
        "forecasting. Run the service locally to use your own data.";
    }
  }
  document.body.classList.add("static-mode");
}

/* ------------------------------------------------------------- bootstrap */

async function loadService() {
  const node = el("service-state");
  const text = node.querySelector(".state-text");
  node.classList.remove("ready", "degraded", "down");

  try {
    const health = await getJSON("/health");
    state.health = health;
    if (health.models_loaded && health.data_connected) {
      node.classList.add("ready");
      text.textContent = "System ready";
    } else {
      node.classList.add("degraded");
      text.textContent = "Not ready";
      showAlert(
        "The system is running but cannot forecast yet.\n\nStill needed:\n  " +
          (health.missing.join("\n  ") || "unknown") +
          "\n\nTrain the models, then restart the service.",
        "info"
      );
    }
  } catch (err) {
    node.classList.add("down");
    text.textContent = "Unreachable";
    showAlert(`Could not reach the service: ${err.message}`);
    renderLineage();
    return false;
  }

  try {
    const range = await getJSON("/dates");
    state.dates = range;
    if (range.synthetic) el("synthetic-banner").classList.remove("hidden");
    // In static mode the picker is already pinned to the snapshot's one date;
    // the full range would advertise dates this deployment cannot serve.
    if (range.available && !state.staticMode) {
      const input = el("date");
      input.min = range.start;
      input.max = range.end;
      input.value = range.end;
      el("date-hint").textContent = `${range.start} to ${range.end} · ${range.n_dates} days`;
    }
    if (range.synthetic) el("brand-version").textContent = "Synthetic dataset";
  } catch (err) {
    el("date-hint").textContent = "No dates available";
  }

  try {
    const districts = await getJSON("/districts");
    state.districtSource = districts;
    const list = el("district-list");
    list.innerHTML = "";
    districts.districts.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      list.appendChild(option);
    });
    if (districts.districts.length) el("location").value = districts.districts[0];
  } catch (err) {
    /* Not fatal -- the user can still type a district name. */
  }

  renderLineage();
  renderHealth();
  renderChain(null);
  loadVerification();
  return true;
}

/* ------------------------------------------------------ hero comparison */

function renderHero(data, requestedDate) {
  const raw = data.raw_forecast_mm;
  const corrected = data.corrected_forecast_mm;
  const observed = data.observed_mm;
  const hasRaw = raw !== null && raw !== undefined;
  const hasObserved = observed !== null && observed !== undefined;

  // 1. What the stored forecast already said.
  el("raw-mm").textContent = num(raw);
  el("raw-note").textContent = hasRaw
    ? "Already in your database"
    : "No stored forecast for this cell";

  // 2. What we changed it to.
  el("metric-mode").textContent = data.routing === "soft" ? "Our prediction" : "Single specialist";
  el("corrected-mm").textContent = num(corrected);

  const band = data.interval;
  const rangeNode = el("corrected-range");
  if (band) {
    const nominal = Math.round((band.nominal_coverage || 0) * 100);
    // A band whose measured coverage missed the mark is a false reassurance,
    // so it is labelled rather than shown bare.
    rangeNode.textContent = band.calibrated === false
      ? `${fmt(band.low_mm)}–${fmt(band.high_mm)} · not well calibrated`
      : `${fmt(band.low_mm)}–${fmt(band.high_mm)} · ${nominal}% confidence band`;
  } else {
    rangeNode.textContent = "";
  }

  const delta = el("corrected-delta");
  if (hasRaw) {
    const d = corrected - raw;
    delta.className = `fig-d ${d >= 0 ? "up" : "down"}`;
    delta.textContent = `${d >= 0 ? "+" : "−"}${fmt(Math.abs(d), 1)} mm vs stored`;
  } else {
    delta.className = "fig-d";
    delta.textContent = "";
  }

  // 3. What actually fell -- only ever shown for a measured day.
  el("observed-mm").textContent = hasObserved ? num(observed) : "—";
  const obsNote = el("observed-note");
  if (!hasObserved) {
    obsNote.className = "fig-d";
    obsNote.textContent = "Not measured yet — this day is a forecast";
  } else if (hasRaw) {
    const before = Math.abs(raw - observed);
    const after = Math.abs(corrected - observed);
    obsNote.className = "fig-d";
    obsNote.textContent = after < before
      ? `We closed ${fmt(before - after, 1)} mm of the gap`
      : `We widened the gap by ${fmt(after - before, 1)} mm`;
  } else {
    obsNote.className = "fig-d";
    obsNote.textContent = "Measured";
  }

  // The connectors carry the arithmetic so the reader does not have to.
  el("flow-delta").textContent = hasRaw
    ? `${corrected - raw >= 0 ? "+" : "−"}${fmt(Math.abs(corrected - raw), 1)} mm`
    : "";
  el("flow-gap").textContent = hasObserved
    ? `${fmt(Math.abs(corrected - observed), 1)} mm off`
    : "not measured";

  // Context, not a fourth outcome.
  const anomaly = data.anomaly;
  const context = el("anomaly-line");
  if (anomaly) {
    const wetter = anomaly.anomaly_mm >= 0;
    const big = Math.abs(anomaly.anomaly_pct || 0) >= 100;
    context.className = `hero-note${big ? " alarm" : ""}`;
    context.innerHTML =
      `A normal day here this month brings about <b>${fmt(anomaly.climatology_mm, 1)} mm</b>. ` +
      `This is <b>${fmt(Math.abs(anomaly.anomaly_mm), 1)} mm ${wetter ? "more" : "less"}</b>` +
      (anomaly.anomaly_pct !== null && anomaly.anomaly_pct !== undefined
        ? ` than usual (${anomaly.anomaly_pct >= 0 ? "+" : ""}${Math.round(anomaly.anomaly_pct)}%).`
        : " than usual.");
  } else {
    context.className = "hero-note";
    context.textContent = "";
  }
}

/* --------------------------------------------- confidence + weather type */

function renderRegime(data) {
  const blend = data.regime_blend || [];
  const names = blend.map((c) => regimeInfo(c.regime).short);
  el("regime-label").textContent = names.join(" + ") || regimeInfo(data.regime).short;

  const confidence = data.regime_confidence || 0;
  el("regime-confidence").textContent = Math.round(confidence * 100);

  // Rounded-rect perimeter: 2*(92-40) straight + 2*pi*20 arcs.
  const perimeter = 229.7;
  el("conf-fill").style.strokeDashoffset = String(perimeter * (1 - confidence));

  el("regime-note").textContent = blend.length > 1
    ? "Transition zone — two types blended."
    : regimeInfo(blend.length ? blend[0].regime : data.regime).description;

  const named = new Set(blend.map((c) => c.regime));
  const container = el("regime-bars");
  container.innerHTML = "";
  REGIME_ORDER
    .map((name) => [name, data.regime_probability[name] || 0])
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, value]) => {
      const info = regimeInfo(name);
      const row = document.createElement("div");
      row.className = named.has(name) ? "bar-row lead" : "bar-row";
      row.innerHTML =
        `<span class="bar-name" title="${info.description}">${info.short}</span>` +
        `<span class="bar-track"><span class="bar-fill" style="width:${(value * 100).toFixed(1)}%;` +
        `background:${info.colour}"></span></span>` +
        `<span class="bar-val">${pct(value)}</span>`;
      container.appendChild(row);
    });

  // Real coordinates, straight from the API's district centroid.
  el("district-lat").textContent = data.centroid_lat === null || data.centroid_lat === undefined
    ? "—" : `${fmt(data.centroid_lat, 2)}°N`;
  el("district-lon").textContent = data.centroid_lon === null || data.centroid_lon === undefined
    ? "—" : `${fmt(data.centroid_lon, 2)}°E`;
  el("district-cells").textContent = String(data.n_grid_cells);

  renderCategoryScale(data.corrected_forecast_mm);

  const confident = blend.length === 1 && blend[0].probability >= 0.8;
  el("routing-note").textContent = confident
    ? "One specialist model handled the correction."
    : "Not certain which type this is, so the specialists were blended.";
}

/* The IMD daily rainfall bands the whole pipeline is scored against. These are
 * evaluation categories, not an agency warning scale -- the panel says so. */
const CATEGORY_BANDS = [
  { key: "no_rain", label: "Dry", lo: 0, hi: 2.5 },
  { key: "light", label: "Light", lo: 2.5, hi: 15.6 },
  { key: "moderate", label: "Moderate", lo: 15.6, hi: 64.5 },
  { key: "heavy", label: "Heavy", lo: 64.5, hi: 115.6 },
  { key: "very_heavy", label: "Very heavy", lo: 115.6, hi: 204.4 },
  { key: "extremely_heavy", label: "Extreme", lo: 204.4, hi: Infinity },
];

function categoryFor(value) {
  if (!Number.isFinite(value)) return null;
  return CATEGORY_BANDS.find((b) => value >= b.lo && value < b.hi) || CATEGORY_BANDS[CATEGORY_BANDS.length - 1];
}

function renderCategoryScale(value) {
  const container = el("category-scale");
  if (!container) return;
  const active = categoryFor(value);
  container.innerHTML = CATEGORY_BANDS.map((b) => {
    const on = active && b.key === active.key;
    const upper = b.hi === Infinity ? "+" : `–${fmt(b.hi, 1)}`;
    return `<span class="cat${on ? " on" : ""}" title="${fmt(b.lo, 1)}${upper} mm">` +
      `<i></i>${b.label}</span>`;
  }).join("") +
    '<span class="cat-note">Project evaluation categories, not official warning levels.</span>';
}

/* ------------------------------------------------ exceedance probability */

function renderProbabilities(data) {
  const container = el("probability-bars");
  container.innerHTML = "";
  // Prefer the richer list, which carries the caller's own threshold.
  const entries = (data.threshold_probabilities || []).length
    ? data.threshold_probabilities.map((t) => ({
        name: t.name, mm: t.threshold_mm, value: t.probability,
        custom: t.name === "custom", interpolated: t.interpolated,
      }))
    : Object.entries(data.heavy_rain_probability || {}).map(([name, value]) => ({
        name, mm: THRESHOLD_MM[name], value, custom: false, interpolated: false,
      }));

  if (!entries.length) {
    container.innerHTML = '<p class="row-note">The risk model is not loaded, so no percentages can be shown.</p>';
    return;
  }

  entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = entry.custom ? "prob-row custom" : "prob-row";
    const label = entry.custom ? "Your limit" : (THRESHOLD_LABEL[entry.name] || entry.name);
    row.innerHTML =
      `<span class="prob-name">${label}` +
      // An interpolated estimate is never presented as a fitted model.
      `<small>over ${fmt(entry.mm, 1)} mm${entry.interpolated ? " · estimated" : ""}</small></span>` +
      `<span class="prob-val">${pct(entry.value)}</span>` +
      `<span class="prob-track"><span class="prob-fill" style="width:${
        (entry.value * 100).toFixed(1)}%"></span></span>`;
    container.appendChild(row);
  });
}

function renderWarning(data) {
  const card = el("warning-card");
  const level = String(data.warning_level || "none").toLowerCase();
  const text = WARNING_TEXT[level] || WARNING_TEXT.none;
  card.className = `panel advisory level-${level}`;
  el("warning-level").textContent = text.headline;
  el("warning-basis").textContent = text.detail;
}

/* ------------------------------------------------------------ attribution */

function renderShap(containerId, contributions, unit) {
  const container = el(containerId);
  container.innerHTML = "";
  if (!contributions || !contributions.length) {
    container.innerHTML = '<p class="row-note">No explanation available.</p>';
    return;
  }
  const peak = Math.max(...contributions.map((c) => Math.abs(c.shap_value))) || 1;

  contributions.forEach((c) => {
    const frac = Math.abs(c.shap_value) / peak;
    const positive = c.shap_value >= 0;
    const half = (frac * 50).toFixed(1);
    // With a unit the number is meaningful. Without one (classifier evidence,
    // in log-odds) the bar's side and length carry it instead.
    const value = unit
      ? `${positive ? "+" : "−"}${fmt(Math.abs(c.shap_value), 1)}`
      : (positive ? "supports" : "against");

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      `<span class="bar-num ${positive ? "pos" : "neg"}">${value}</span>` +
      `<span class="bar-track"><span class="bar-mid"></span>` +
      `<span class="bar-fill ${positive ? "pos" : "neg"}" style="width:${half}%"></span></span>` +
      `<span class="bar-val" title="${c.feature}">${featureLabel(c.feature)}` +
      `<i class="bar-var">${c.feature}</i></span>`;
    container.appendChild(row);
  });
}

function renderBiasExplanation(data) {
  const detail = data.bias_explanation;
  const note = el("bias-shap-note");
  if (!detail) {
    renderShap("bias-shap", null, "");
    note.textContent = "";
    return;
  }
  renderShap("bias-shap", detail.top_features, " mm");

  const generic = detail.regime === "__fallback__";
  const specialist = generic
    ? "a general-purpose model"
    : `the ${regimeInfo(detail.regime).short.toLowerCase()} specialist`;
  const verb = detail.predicted_bias_mm >= 0 ? "added" : "removed";
  note.textContent =
    `Wettest cell: ${specialist} ${verb} ${fmt(Math.abs(detail.predicted_bias_mm), 1)} mm, ` +
    `${fmt(detail.raw_mm, 1)} → ${fmt(detail.corrected_mm, 1)} mm`;
}

/* ------------------------------------------------------------ trend chart */

function renderTrendChart(payload) {
  const canvas = el("trend-canvas");
  const empty = el("chart-empty");
  const steps = (payload && payload.steps) || [];

  if (steps.length < 2) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    empty.textContent = steps.length
      ? "Only one day in this window — widen the time span."
      : "No days available for this window.";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(rect.width, 320);
  const H = Math.max(rect.height, 200);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const padL = 44, padR = 12, padT = 14, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const cGrid = themeColour("--c-grid", "rgba(255,255,255,.05)");
  const cGridNow = themeColour("--c-grid-now", "rgba(127,209,222,.28)");
  const cAxis = themeColour("--c-axis", "#7d8894");
  const cCore = themeColour("--c-point-core", "#0b1016");
  const glow = themeColour("--glow", "1") !== "0";

  const series = [
    { key: "corrected_forecast_mm", theme: "--cyan", fallback: "#7fd1de", dash: null, lead: true },
    { key: "raw_forecast_mm", theme: "--indigo", fallback: "#8b93c9", dash: null, lead: false },
    { key: "observed_mm", theme: "--brass", fallback: "#c9a227", dash: [4, 4], lead: false },
  ];

  const all = [];
  series.forEach((s) => steps.forEach((st) => {
    const v = st[s.key];
    if (v !== null && v !== undefined && Number.isFinite(v)) all.push(v);
  }));
  const maxV = Math.max(...all, 1);
  const step = maxV > 200 ? 50 : maxV > 80 ? 20 : maxV > 30 ? 10 : 5;
  const top = Math.ceil(maxV / step) * step;

  const X = (i) => padL + (i / (steps.length - 1)) * plotW;
  const Y = (v) => padT + plotH - (v / top) * plotH;

  ctx.font = "10px Inter, sans-serif";
  ctx.strokeStyle = cGrid;
  ctx.lineWidth = 1;
  ctx.fillStyle = cAxis;
  for (let v = 0; v <= top; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(String(v), 8, y + 3);
  }
  steps.forEach((st, i) => {
    const x = Math.round(X(i)) + 0.5;
    if (st.offset_days === 0) {
      ctx.strokeStyle = cGridNow;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
    }
    const label = shortDate(st.date);
    const halfW = ctx.measureText(label).width / 2;
    const lx = Math.min(Math.max(x - halfW, 2), W - halfW * 2 - 2);
    ctx.fillStyle = st.offset_days === 0 ? themeColour("--cyan", "#7fd1de") : cAxis;
    ctx.fillText(label, lx, H - 9);
  });

  series.forEach((s) => {
    const points = steps
      .map((st, i) => ({ v: st[s.key], x: X(i), y: Y(st[s.key]) }))
      .filter((p) => p.v !== null && p.v !== undefined && Number.isFinite(p.v));
    if (points.length < 2) return;
    const colour = themeColour(s.theme, s.fallback);

    // Our corrected series carries a soft area wash: it is the answer the page
    // exists to give, and it should read first.
    if (s.lead) {
      ctx.save();
      const wash = ctx.createLinearGradient(0, padT, 0, padT + plotH);
      wash.addColorStop(0, hexToRGBA(colour, glow ? 0.18 : 0.14));
      wash.addColorStop(1, hexToRGBA(colour, 0));
      ctx.beginPath();
      ctx.moveTo(points[0].x, padT + plotH);
      ctx.lineTo(points[0].x, points[0].y);
      catmullRom(ctx, points);
      ctx.lineTo(points[points.length - 1].x, padT + plotH);
      ctx.closePath();
      ctx.fillStyle = wash;
      ctx.fill();
      ctx.restore();
    }

    ctx.save();
    ctx.setLineDash(s.dash || []);
    ctx.strokeStyle = colour;
    ctx.lineWidth = s.lead ? 2 : 1.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    catmullRom(ctx, points);
    ctx.stroke();
    ctx.restore();

    if (s.dash) return;
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
      ctx.fillStyle = cCore;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = colour;
      ctx.stroke();
    });
  });

  el("chart-meta").textContent =
    `Daily steps · ${steps.length} days · peak ${fmt(maxV, 1)} mm`;
  el("chart-title").textContent =
    `Rainfall Trend (${shortDate(steps[0].date)}–${shortDate(steps[steps.length - 1].date)})`;
}

function renderSteps(payload) {
  const container = el("timeline");
  container.innerHTML = "";
  (payload.steps || []).forEach((step) => {
    const button = document.createElement("button");
    button.className = `step${step.offset_days === 0 ? " now" : ""}`;
    button.innerHTML =
      `<span class="step-v">${num(step.corrected_forecast_mm)} mm</span>` +
      `<span class="step-d">${shortDate(step.date)}</span>` +
      `<span class="step-t">${plainRegimeLabel(step.regime_label)}</span>` +
      (step.regime_changed ? '<span class="step-f">type shift</span>' : "");
    button.addEventListener("click", () => {
      el("date").value = step.date;
      run();
    });
    container.appendChild(button);
  });

  const transitions = el("transitions");
  if (payload.transitions && payload.transitions.length) {
    transitions.classList.remove("hidden");
    transitions.innerHTML = payload.transitions.map((line) => {
      const readable = line.replace(
        /^(.+?) to (.+?) on (.+)$/,
        (_, from, to, when) => `${plainRegimeLabel(from)} → ${plainRegimeLabel(to)} on ${prettyDate(when)}`
      );
      return `<p>${readable}</p>`;
    }).join("");
  } else {
    transitions.classList.add("hidden");
  }
}

function renderTimeline(payload) {
  state.lastTimeline = payload;
  renderTrendChart(payload);
  renderSteps(payload);
}

/* ------------------------------------------------------------------- maps */

function drawPanel(canvas, values, lats, lons, ramp, scale) {
  const ctx = canvas.getContext("2d");
  canvas.width = lons.length;
  canvas.height = lats.length;
  const image = ctx.createImageData(lons.length, lats.length);
  const blank = isLight() ? [247, 245, 241] : [11, 16, 22];

  for (let row = 0; row < lats.length; row += 1) {
    // Latitude ascends northward but canvas rows descend, so flip.
    const sourceRow = lats.length - 1 - row;
    for (let col = 0; col < lons.length; col += 1) {
      const value = values[sourceRow * lons.length + col];
      const offset = (row * lons.length + col) * 4;
      if (value === null || value === undefined || !Number.isFinite(value)) {
        image.data[offset] = blank[0];
        image.data[offset + 1] = blank[1];
        image.data[offset + 2] = blank[2];
        image.data[offset + 3] = 255;
        continue;
      }
      const colour = sampleRamp(ramp, scale(value));
      image.data[offset] = colour[0];
      image.data[offset + 1] = colour[1];
      image.data[offset + 2] = colour[2];
      image.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

/* The big map panel. Hovering reads the cell under the cursor straight out of
 * the grid payload, so the tooltip reports real coordinates and real values --
 * no interpolation, no invented precision. */
function renderHeroMap(grid) {
  const canvas = el("map-hero");
  if (!canvas || !grid) return;
  const layer = state.mapLayer || "corrected";
  const unavailable = layer === "observed" && !grid.observed_available;

  const scales = gridScales(grid);
  const scratch = document.createElement("canvas");
  if (unavailable) {
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.clientWidth || 600;
    canvas.height = canvas.clientHeight || 320;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    el("map-tip").textContent = "This day has no measurement yet.";
    return;
  }

  if (layer === "difference") {
    drawPanel(scratch, grid.panels.difference, grid.lats, grid.lons, diffRamp(),
      (v) => diffScale(v, scales.diffPeak));
  } else {
    drawPanel(scratch, grid.panels[layer], grid.lats, grid.lons, rainRamp(),
      (v) => rainScale(v, scales.rainMax));
  }

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(Math.round((rect.width || 600) * dpr), 1);
  canvas.height = Math.max(Math.round((rect.height || 320) * dpr), 1);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(scratch, 0, 0, canvas.width, canvas.height);
}

function gridScales(grid) {
  const stats = {};
  (grid.panel_stats || []).forEach((p) => { stats[p.name] = p; });
  const rainPanels = ["raw", "corrected"].concat(grid.observed_available ? ["observed"] : []);
  const rainMax = Math.max(...rainPanels.map((n) => (stats[n] && stats[n].max_value) || 0), 1);
  const d = stats.difference || {};
  const diffPeak = Math.max(Math.abs(d.min_value || 0), Math.abs(d.max_value || 0), 0.5);
  return { rainMax, diffPeak };
}

/* Translate a pointer position into the grid cell beneath it. */
function cellAt(grid, canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const fx = (clientX - rect.left) / rect.width;
  const fy = (clientY - rect.top) / rect.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
  const col = Math.min(Math.floor(fx * grid.lons.length), grid.lons.length - 1);
  // Rows are drawn north-to-south; latitudes ascend, so flip back.
  const rowFromTop = Math.min(Math.floor(fy * grid.lats.length), grid.lats.length - 1);
  const row = grid.lats.length - 1 - rowFromTop;
  return { row, col, index: row * grid.lons.length + col };
}

function describeCell(grid, hit) {
  if (!hit) return "";
  const lat = grid.lats[hit.row];
  const lon = grid.lons[hit.col];
  const val = (name) => {
    const arr = (grid.panels || {})[name];
    const v = arr ? arr[hit.index] : null;
    return v === null || v === undefined || !Number.isFinite(v) ? "—" : `${fmt(v, 1)} mm`;
  };
  const type = (grid.regimes || [])[hit.index];
  return `<b>${fmt(lat, 2)}°N, ${fmt(lon, 2)}°E</b>` +
    `<span>Existing <i>${val("raw")}</i></span>` +
    `<span>Our correction <i>${val("corrected")}</i></span>` +
    (grid.observed_available ? `<span>Measured <i>${val("observed")}</i></span>` : "") +
    (type ? `<span>Weather type <i>${regimeInfo(type).short}</i></span>` : "");
}

const PANEL_TEXT = {
  raw: { title: "Weather model", note: "Before any correction." },
  corrected: { title: "Our correction", note: "The same day, corrected." },
  observed: { title: "Measured", note: "What actually fell." },
  difference: { title: "Change applied", note: "Blue added rain, red removed it." },
};

function renderMaps(grid) {
  state.lastGrid = grid;
  const container = el("map-grid");
  container.innerHTML = "";

  const showObserved = el("show-observed").checked;
  const wanted = ["raw", "corrected"];
  if (showObserved && grid.observed_available) wanted.push("observed");
  wanted.push("difference");

  const statsByName = {};
  (grid.panel_stats || []).forEach((s) => { statsByName[s.name] = s; });

  // One shared rainfall scale across the panels. Independent scales would make
  // any correction look dramatic.
  const rainPanels = ["raw", "corrected"].concat(grid.observed_available ? ["observed"] : []);
  const rainMax = Math.max(
    ...rainPanels.map((n) => (statsByName[n] && statsByName[n].max_value) || 0), 1
  );
  const diffStats = statsByName.difference || {};
  const diffPeak = Math.max(
    Math.abs(diffStats.min_value || 0), Math.abs(diffStats.max_value || 0), 0.5
  );

  wanted.forEach((name) => {
    if (!statsByName[name]) return;
    const text = PANEL_TEXT[name];
    const cell = document.createElement("div");
    cell.className = "map-cell";
    const heading = document.createElement("h5");
    heading.textContent = text.title;
    const canvas = document.createElement("canvas");
    const note = document.createElement("p");
    note.textContent = text.note;

    if (name === "difference") {
      drawPanel(canvas, grid.panels[name], grid.lats, grid.lons, diffRamp(),
        (v) => diffScale(v, diffPeak));
    } else {
      drawPanel(canvas, grid.panels[name], grid.lats, grid.lons, rainRamp(),
        (v) => rainScale(v, rainMax));
    }
    cell.append(heading, canvas, note);
    container.appendChild(cell);
  });

  renderMapLegend(rainMax, diffPeak);
  renderHeroMap(grid);

  const [latMin, latMax, lonMin, lonMax] = grid.bbox.length === 4 ? grid.bbox : [0, 0, 0, 0];
  el("map-status").textContent =
    `Extent ${fmt(latMin)}–${fmt(latMax)}°N, ${fmt(lonMin)}–${fmt(lonMax)}°E · ` +
    `${grid.lons.length} × ${grid.lats.length} cells` +
    (grid.observed_available ? "" : " · no measurement for this date yet");
}

function renderMapLegend(rainMax, diffPeak) {
  const legend = el("map-legend");
  legend.innerHTML = "";
  const build = (label, ramp, caption) => {
    const item = document.createElement("div");
    item.className = "maplegend-item";
    const strip = document.createElement("span");
    strip.style.display = "inline-flex";
    for (let i = 0; i < 18; i += 1) {
      const swatch = document.createElement("span");
      swatch.style.cssText = `width:5px;height:8px;background:${rgb(sampleRamp(ramp, i / 17))}`;
      strip.appendChild(swatch);
    }
    item.append(strip, document.createTextNode(` ${label} ${caption}`));
    legend.appendChild(item);
  };
  build("Rain", rainRamp(), `0 → ${fmt(rainMax, 0)} mm`);
  build("Change", diffRamp(), `−${fmt(diffPeak, 0)} → +${fmt(diffPeak, 0)} mm`);
}

/* ---------------------------------------------------------------- drivers */

function renderDrivers(payload) {
  const container = el("drivers");
  container.innerHTML = "";
  const drivers = payload.drivers || [];
  if (!drivers.length) {
    container.innerHTML = '<p class="row-note">No drivers could be computed for this day.</p>';
    el("drivers-status").textContent = "";
    return;
  }

  const peak = Math.max(...drivers.map((d) => d.mean_abs_contribution_mm)) || 1;
  const DIRECTION = { up: "adds rain", down: "removes rain", mixed: "both ways" };
  drivers.slice(0, 6).forEach((driver) => {
    const row = document.createElement("div");
    row.className = "driver";
    row.innerHTML =
      `<span class="driver-name" title="${driver.feature}">${featureLabel(driver.feature)}</span>` +
      `<span class="driver-track"><span class="driver-fill" style="width:${
        (driver.mean_abs_contribution_mm / peak * 100).toFixed(1)}%"></span></span>` +
      `<span class="driver-val ${driver.direction}">±${fmt(driver.mean_abs_contribution_mm, 2)} mm</span>`;
    row.title = DIRECTION[driver.direction] || driver.direction;
    container.appendChild(row);
  });

  const dominant = payload.dominant_regime ? regimeInfo(payload.dominant_regime) : null;
  el("drivers-status").textContent =
    (dominant ? `Mostly ${dominant.short.toLowerCase()} · ` : "") +
    `${payload.n_cells_sampled} cells${payload.sampled ? " (sampled)" : ""}`;
}

/* --------------------------------------------------------- data lineage */

function renderLineage() {
  const tbody = document.querySelector("#lineage-table tbody");
  const health = state.health;
  const dates = state.dates;
  const districts = state.districtSource;
  const grid = state.lastGrid;
  const missing = (health && health.missing) || [];
  const has = (needle) => !missing.some((m) => m.toLowerCase().includes(needle));

  const resolution = grid && grid.lons && grid.lons.length
    ? `${grid.lons.length} × ${grid.lats.length} cells`
    : "—";

  const rows = [
    { dot: "var(--cyan)", source: "Joined analysis table", id: "analysis_table",
      res: resolution, freq: "Daily (24 h)", ok: !!(health && health.data_connected), okLabel: "Synced" },
    { dot: "var(--indigo)", source: "District boundaries",
      id: districts ? `source: ${districts.source}` : "districts",
      res: districts ? `${districts.count} districts` : "—", freq: "Static",
      ok: !!(health && health.districts_available), okLabel: "Loaded" },
    { dot: "var(--brass)", source: "Weather-type classifier", id: "regime_classifier.joblib",
      res: `${REGIME_ORDER.length} classes`, freq: "On retrain", ok: has("classifier"), okLabel: "Loaded" },
    { dot: "var(--cyan)", source: "Per-type correction models", id: "bias_correction.joblib",
      res: "Residual, mm", freq: "On retrain", ok: has("bias-correction"), okLabel: "Loaded" },
    { dot: "var(--brass)", source: "Heavy-rain probability models", id: "heavy_rain_probability.joblib",
      res: `${Object.keys(THRESHOLD_MM).length} thresholds`, freq: "On retrain",
      ok: has("probability"), okLabel: "Calibrated" },
    { dot: "var(--indigo)", source: "Verification report", id: "verification_report.json",
      res: state.lastReport ? `${Object.keys(state.lastReport.models || {}).length} methods` : "—",
      freq: "On retrain", ok: !!state.lastReport, okLabel: "Available" },
  ];

  tbody.innerHTML = rows.map((r) =>
    `<tr><td><span class="src"><span class="src-dot" style="background:${r.dot}"></span>${r.source}</span></td>` +
    `<td>${r.id}</td><td>${r.res}</td><td>${r.freq}</td>` +
    `<td><span class="tag ${r.ok ? "tag-ok" : "tag-bad"}">${r.ok ? r.okLabel : "Missing"}</span></td></tr>`
  ).join("");

  el("lineage-note").textContent = dates && dates.available
    ? `Coverage ${dates.start} → ${dates.end} · ${dates.n_dates} days` +
      (dates.synthetic ? " · synthetic demo data" : "")
    : "No dataset connected. Add your files and run the training pipeline.";
}

function renderHealth() {
  const container = el("health-grid");
  const health = state.health;
  if (!health) {
    container.innerHTML = '<p class="row-note">Service unreachable.</p>';
    return;
  }
  const dates = state.dates || {};
  const cells = [
    ["Models", health.models_loaded ? "Loaded" : "Not trained"],
    ["Data", health.data_connected ? "Connected" : "Missing"],
    ["Districts", health.districts_available ? "Available" : "Missing"],
    ["Artifact dir", health.artifact_dir],
    ["Coverage", dates.available ? `${dates.start} → ${dates.end}` : "—"],
    ["Dataset", dates.synthetic ? "Synthetic (demo)" : "Connected dataset"],
  ];
  container.innerHTML = cells.map(([k, v]) =>
    `<div class="health-cell"><span class="k">${k}</span><span class="v">${v}</span></div>`
  ).join("");
  if (health.missing && health.missing.length) {
    container.innerHTML +=
      `<div class="health-cell" style="grid-column:1/-1">` +
      `<span class="k">Still needed</span><span class="v">${health.missing.join("<br>")}</span></div>`;
  }
}

const CHAIN_STAGES = [
  { key: "input", title: "Weather model in" },
  { key: "features", title: "Multimodal features" },
  { key: "engine", title: "Weather-type engine" },
  { key: "router", title: "Router" },
  { key: "correct", title: "Specialist correction" },
  { key: "risk", title: "Exceedance odds" },
  { key: "district", title: "District output" },
  { key: "verify", title: "Scorecard" },
];

function renderChain(data, targetId = "chain") {
  const container = el(targetId);
  if (!container) return;
  const values = {};
  if (data) {
    const detail = data.bias_explanation;
    const heavy = (data.heavy_rain_probability || {}).heavy;
    values.input = mm(data.raw_forecast_mm);
    values.features = detail && detail.top_features ? `${detail.top_features.length} shown` : "built";
    values.engine = `${plainRegimeLabel(data.regime_label)} · ${pct(data.regime_confidence)}`;
    values.router = data.routing === "soft" ? "blended" : "single";
    values.correct = detail
      ? `${detail.predicted_bias_mm >= 0 ? "+" : "−"}${mm(Math.abs(detail.predicted_bias_mm))}`
      : null;
    values.risk = heavy === undefined ? null : `${pct(heavy)} heavy`;
    values.district = `${mm(data.corrected_forecast_mm)} · ${data.n_grid_cells} cells`;
  }
  const report = state.lastReport;
  if (report && report.models && report.models[OUR_MODEL]) {
    const rmse = metricsFor(report.models[OUR_MODEL]).rmse;
    if (Number.isFinite(rmse)) values.verify = `${mm(rmse)} typical error`;
  }

  container.innerHTML = CHAIN_STAGES.map((stage, i) => {
    const value = values[stage.key];
    return `<li class="${value ? "" : "off"}">` +
      `<span class="n">${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="t">${stage.title}</span>` +
      `<span class="v">${value || "waiting"}</span></li>`;
  }).join("");
}

/* ---------------------------------------------------------- verification */

function metricsFor(entry) {
  const overall = (entry && entry.overall) || {};
  const continuous = overall.continuous || {};
  const heavy = (overall.categorical && overall.categorical.heavy) || {};
  const fss = (overall.fss && overall.fss.heavy) || {};
  return {
    rmse: continuous.rmse, bias: continuous.bias, mae: continuous.mae,
    correlation: continuous.correlation,
    pod: heavy.pod, far: heavy.far, csi: heavy.csi, ets: heavy.ets,
    fss: fss.window_3,
  };
}

/* Plain label, plain meaning, and which direction is better. Kept out of the
 * default view -- these live behind the technical disclosure. */
const METRIC_INFO = {
  rmse: { label: "Typical error", better: "lower", plain: "How far off a typical forecast is.", technical: "RMSE — root mean squared error." },
  bias: { label: "Over/under-forecasting", better: "zero", plain: "Whether it leans wet or dry. Zero is best.", technical: "Mean bias." },
  mae: { label: "Average miss", better: "lower", plain: "The average size of the error, ignoring direction.", technical: "MAE — mean absolute error." },
  correlation: { label: "Tracks reality", better: "higher", plain: "Whether wet days come out wet. 1.0 is perfect.", technical: "Pearson correlation." },
  pod: { label: "Dangerous rain caught", better: "higher", plain: "Of the genuinely heavy events, how many it flagged.", technical: "POD — probability of detection." },
  far: { label: "False alarms", better: "lower", plain: "Of everything flagged heavy, how much never happened.", technical: "FAR — false alarm ratio." },
  csi: { label: "Overall heavy-rain score", better: "higher", plain: "Catching real events and avoiding false alarms combined.", technical: "CSI — critical success index." },
  ets: { label: "Skill beyond luck", better: "higher", plain: "The same, minus credit for chance hits.", technical: "ETS — equitable threat score." },
  fss: { label: "Rain in the right place", better: "higher", plain: "Whether rain lands in roughly the right spot.", technical: "FSS — fractions skill score, 3-cell neighbourhood." },
};

const INTENSITY_LABELS = {
  no_rain: "Dry", light: "Light rain", moderate: "Moderate rain",
  heavy: "Heavy rain", very_heavy: "Very heavy rain", extremely_heavy: "Extreme rain",
};
const INTENSITY_ORDER = ["no_rain", "light", "moderate", "heavy", "very_heavy", "extremely_heavy"];

/* The report no longer has a screen of its own, but it still feeds the lineage
 * row and the chain's final stage, and Export Report serves the whole thing. */
async function loadVerification() {
  let payload;
  try {
    payload = await getJSON("/verification-report");
  } catch (err) {
    return;
  }
  if (!payload.available || !payload.report) return;
  state.lastReport = payload.report;
  renderLineage();
  renderChain(state.lastPrediction);
}

/* -------------------------------------------------------------- analytics
 *
 * Every figure here is read out of the saved verification report. Nothing is
 * restated as an "accuracy" gain: an error reduction is what was measured, so
 * an error reduction is what is shown.
 */

function renderAnalytics() {
  const report = state.lastReport;
  const cards = el("metric-cards");
  if (!report) {
    cards.innerHTML = '<p class="row-note">No verification report yet. Run the ' +
      "training pipeline, then restart the service.</p>";
    return;
  }

  el("verification-period").textContent =
    `Held out ${report.test_period.start} → ${report.test_period.end} · ` +
    `${Number(report.n_test_rows).toLocaleString()} rows`;

  const ours = metricsFor(report.models[OUR_MODEL]);
  const raw = metricsFor(report.models[RAW_MODEL]);
  const drop = Number.isFinite(raw.rmse) && raw.rmse
    ? ((raw.rmse - ours.rmse) / raw.rmse) * 100 : null;

  const tiles = [
    ["Existing forecast error", fmt(raw.rmse, 2), "mm typical error", ""],
    ["Our error", fmt(ours.rmse, 2), "mm typical error", "count--none"],
    ["Error reduction", drop === null ? "—" : `${fmt(drop, 2)}%`,
      "lower error than the existing forecast", "count--none"],
    ["Tracks reality", fmt(ours.correlation, 3), "1.0 would be perfect", ""],
    ["Over/under-forecasting", fmt(ours.bias, 2), "mm — zero is best", ""],
    ["Average miss", fmt(ours.mae, 2), "mm", ""],
  ];
  cards.innerHTML = tiles.map(([k, v, sub, cls]) =>
    `<div class="count ${cls}"><span class="count-k">${k}</span>` +
    `<span class="count-n">${v}</span><span class="count-sub">${sub}</span></div>`
  ).join("");

  renderRanking(report);
  renderCategoryTable(report);
  renderErrorAnalysis(report);
  renderTechnicalTable(report);
  renderGlossary();
}

function renderRanking(report) {
  const container = el("model-ranking");
  container.innerHTML = "";

  const rows = Object.entries(report.models)
    .map(([id, entry]) => ({ id, metrics: metricsFor(entry) }))
    .filter((r) => Number.isFinite(r.metrics.rmse))
    .sort((a, b) => a.metrics.rmse - b.metrics.rmse);
  if (!rows.length) return;

  const baseline = (rows.find((r) => r.id === RAW_MODEL) || {}).metrics;
  const base = baseline ? baseline.rmse : null;
  const best = rows[0];
  // The bar shows how much error each method removes, not how much is left --
  // sizing by error would make the worst method the longest bar.
  const gains = rows.map((r) => base ? Math.max(((base - r.metrics.rmse) / base) * 100, 0) : 0);
  const bestGain = Math.max(...gains, 1);

  rows.forEach((row, i) => {
    const info = modelInfo(row.id);
    const isOurs = row.id === OUR_MODEL || row.id === "E_regime_residual_probability";
    const vs = base ? ((base - row.metrics.rmse) / base) * 100 : null;
    const node = document.createElement("div");
    node.className = `rank${isOurs ? " ours" : ""}`;
    node.innerHTML =
      `<div class="rank-head"><span class="rank-name">` +
        `<span class="rank-pos">${String(i + 1).padStart(2, "0")}</span>${info.name}` +
        `${row === best ? '<span class="badge">lowest error</span>' : ""}</span>` +
        `<span class="rank-metric">${fmt(row.metrics.rmse, 2)} mm</span></div>` +
      `<p class="rank-role">` +
        (vs !== null && row.id !== RAW_MODEL
          ? `${fmt(Math.abs(vs), 1)}% ${vs >= 0 ? "less" : "more"} error — `
          : "the starting point — ") + info.role + `</p>` +
      `<div class="rank-track"><span class="rank-fill" style="width:${
        Math.max((gains[i] / bestGain) * 100, row.id === RAW_MODEL ? 0 : 3)}%"></span></div>`;
    container.appendChild(node);
  });

  // The honest caveat: when the no-AI ablation places rain better than we do.
  const ours = rows.find((r) => r.id === OUR_MODEL);
  const abl = rows.find((r) => r.id === "C_regime_quantile_mapping");
  if (ours && abl && Number.isFinite(ours.metrics.fss) && abl.metrics.fss > ours.metrics.fss) {
    const p = document.createElement("p");
    p.className = "rank-caveat";
    p.innerHTML = `<strong>Worth knowing:</strong> on placing rain in the right ` +
      `location, the simple weather-type-aware method without AI scores ` +
      `${fmt(abl.metrics.fss, 2)} against our ${fmt(ours.metrics.fss, 2)}.`;
    container.appendChild(p);
  }
}

/* Per-category error. Categories where the correction makes things worse are
 * printed in full -- hiding them would be the single most misleading thing
 * this screen could do. */
function renderCategoryTable(report) {
  const tbody = document.querySelector("#category-table tbody");
  const ours = (report.models[OUR_MODEL] || {}).by || {};
  const raw = (report.models[RAW_MODEL] || {}).by || {};
  const oursB = ours.intensity_bucket || {};
  const rawB = raw.intensity_bucket || {};

  const keys = INTENSITY_ORDER.filter((k) => oursB[k] || rawB[k]);
  if (!keys.length) {
    tbody.innerHTML = '<tr><td colspan="5">No category breakdown in this report.</td></tr>';
    return;
  }

  let improved = 0;
  tbody.innerHTML = keys.map((k) => {
    const o = ((oursB[k] || {}).continuous) || {};
    const r = ((rawB[k] || {}).continuous) || {};
    const change = Number.isFinite(o.rmse) && Number.isFinite(r.rmse) && r.rmse
      ? ((r.rmse - o.rmse) / r.rmse) * 100 : null;
    if (change !== null && change > 0) improved += 1;
    const cls = change === null ? "" : change > 0 ? "good" : "bad";
    const text = change === null ? "—"
      : `${change > 0 ? "−" : "+"}${fmt(Math.abs(change), 2)}% error`;
    return `<tr><td>${INTENSITY_LABELS[k] || k}</td>` +
      `<td class="num">${Number.isFinite(o.n) ? Math.round(o.n).toLocaleString() : "—"}</td>` +
      `<td class="num">${fmt(r.rmse, 2)}</td>` +
      `<td class="num">${fmt(o.rmse, 2)}</td>` +
      `<td class="num ${cls}">${text}</td></tr>`;
  }).join("");

  el("category-note").textContent =
    `Our correction lowers the error in ${improved} of ${keys.length} categories. ` +
    "A negative change means the correction made that category worse.";
}

function renderErrorAnalysis(report) {
  const container = el("error-analysis");
  const ours = metricsFor(report.models[OUR_MODEL]);
  const raw = metricsFor(report.models[RAW_MODEL]);

  const pairs = [
    ["Over/under-forecasting", raw.bias, ours.bias, "zero"],
    ["Tracks reality", raw.correlation, ours.correlation, "higher"],
    ["Average miss", raw.mae, ours.mae, "lower"],
    ["Dangerous rain caught", raw.pod, ours.pod, "higher"],
    ["False alarms", raw.far, ours.far, "lower"],
    ["Rain in the right place", raw.fss, ours.fss, "higher"],
  ];

  container.innerHTML = pairs.map(([label, a, b, better]) => {
    let verdict = "";
    if (Number.isFinite(a) && Number.isFinite(b)) {
      const win = better === "zero" ? Math.abs(b) < Math.abs(a)
        : better === "lower" ? b < a : b > a;
      verdict = win ? "good" : "bad";
    }
    return `<div class="err-row"><span class="err-k">${label}</span>` +
      `<span class="err-a">${fmt(a, 3)}</span>` +
      `<span class="err-arrow" aria-hidden="true">→</span>` +
      `<span class="err-b ${verdict}">${fmt(b, 3)}</span></div>`;
  }).join("");
}

function renderTechnicalTable(report) {
  const order = ["rmse", "bias", "mae", "correlation", "pod", "far", "csi", "ets", "fss"];
  const thead = document.querySelector("#verification-table thead");
  thead.innerHTML = "<tr><th>Method</th>" + order.map((k) => {
    const m = METRIC_INFO[k];
    return `<th class="num" title="${m.technical}">${m.label}</th>`;
  }).join("") + "</tr>";

  const tbody = document.querySelector("#verification-table tbody");
  tbody.innerHTML = Object.entries(report.models).map(([id, entry]) => {
    const m = metricsFor(entry);
    return `<tr><td>${modelInfo(id).name}</td>` + order.map((k) => {
      const v = m[k];
      // An undefined metric is reported as such, never as a flattering zero.
      return Number.isFinite(v)
        ? `<td class="num">${fmt(v, k === "rmse" || k === "bias" || k === "mae" ? 2 : 3)}</td>`
        : `<td class="num null">not defined</td>`;
    }).join("") + "</tr>";
  }).join("");
}

function renderGlossary() {
  const glossary = el("metric-glossary");
  glossary.innerHTML = Object.values(METRIC_INFO).map((m) =>
    `<div><dt>${m.label}</dt><dd>${m.plain}<span class="tech">${m.technical}</span></dd></div>`
  ).join("");
}

/* ------------------------------------------------------------ methodology */

/* The contract's recommended product per slot. Marked as expected inputs, not
 * as what is currently connected -- the loaders read a schema, not a brand. */
const SOURCE_SLOTS = [
  { slot: "Atmospheric predictors", product: "ERA5 (Copernicus CDS)",
    role: "Model input", probe: "analysis" },
  { slot: "Raw forecast", product: "GFS 0.25° (NOAA NOMADS)",
    role: "The forecast being corrected", probe: "analysis" },
  { slot: "Measured rainfall", product: "IMD gridded gauge, or IMERG Final",
    role: "Reference only — never a model input", probe: "analysis" },
  { slot: "District boundaries", product: "Survey of India / data.gov.in",
    role: "Aggregation geometry", probe: "district" },
  { slot: "Terrain and coastline", product: "SRTM / NASADEM",
    role: "Static per-cell model input", probe: "analysis" },
];

function renderSources() {
  const tbody = document.querySelector("#sources-table tbody");
  const health = state.health;
  const missing = (health && health.missing) || [];
  const connected = (probe) => probe === "district"
    ? !!(health && health.districts_available)
    : !!(health && health.data_connected);

  tbody.innerHTML = SOURCE_SLOTS.map((r) => {
    const ok = connected(r.probe);
    return `<tr><td>${r.slot}</td><td>${r.product}</td><td>${r.role}</td>` +
      `<td><span class="tag ${ok ? "tag-ok" : "tag-bad"}">${
        ok ? "Connected" : "Not connected"}</span></td></tr>`;
  }).join("");
}

function renderDatasetStats() {
  const container = el("dataset-stats");
  const dates = state.dates || {};
  const districts = state.districtSource || {};
  const grid = state.lastGrid;
  const report = state.lastReport;

  const cells = [
    ["Coverage", dates.available ? `${dates.start} → ${dates.end}` : "Not connected"],
    ["Days", dates.available ? String(dates.n_dates) : "—"],
    ["Districts", districts.count ? String(districts.count) : "—"],
    ["Grid", grid ? `${grid.lons.length} × ${grid.lats.length} cells` : "—"],
    ["Accumulation", "24 hours (daily)"],
    ["Weather types", String(REGIME_ORDER.length)],
    ["Held-out rows", report ? Number(report.n_test_rows).toLocaleString() : "—"],
    ["Dataset", dates.synthetic ? "Synthetic demo — not real data" : "Connected dataset"],
  ];
  container.innerHTML = cells.map(([k, v]) =>
    `<div class="health-cell"><span class="k">${k}</span><span class="v">${v}</span></div>`
  ).join("");
}

/* ---------------------------------------------------------------- views */

const VIEWS = ["overview", "historical", "risk", "analytics", "methodology", "scenario", "settings"];
const VIEW_TITLE = {
  overview: "Dashboard",
  historical: "Historical Data",
  risk: "Risk Matrix",
  analytics: "Analytics",
  methodology: "Methodology",
  scenario: "Scenario Lab",
  settings: "Settings",
};
const loaded = { watch: null, replay: null };

function currentQuery() {
  return {
    district: el("location").value.trim(),
    date: el("date").value,
    soft: el("soft-routing").checked,
    threshold: parseFloat(el("threshold").value) || null,
    span: parseInt(el("timespan").value, 10) || 2,
  };
}

function showView(name) {
  if (!VIEWS.includes(name)) return;
  VIEWS.forEach((view) => el(`view-${view}`).classList.toggle("hidden", view !== name));
  document.querySelectorAll("#nav button").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.view === name));
  });
  state.view = name;
  el("view-title").textContent = VIEW_TITLE[name] || "Dashboard";

  if (name === "settings") { renderHealth(); renderChain(state.lastPrediction); }
  if (name === "analytics") renderAnalytics();
  if (name === "methodology") { renderChain(state.lastPrediction, "chain-method"); renderSources(); renderDatasetStats(); }

  const query = currentQuery();
  if (!query.date) return;
  if (name === "risk" && loaded.watch !== query.date) loadWatch(query);
  if (name === "historical" && loaded.replay !== "loaded") loadEvents();
  if (name === "scenario") renderScenarioControls();
}

/* ------------------------------------------------------------ risk matrix */

function renderWatch(payload) {
  el("watch-date").textContent = prettyDate(payload.date);

  const summary = el("risk-summary");
  summary.innerHTML = "";
  const counts = payload.counts_by_warning || {};
  [["severe", "Severe"], ["warning", "Warning"], ["watch", "Watch"], ["none", "Clear"]]
    .forEach(([level, label]) => {
      const cell = document.createElement("div");
      cell.className = `count count--${level}`;
      cell.innerHTML = `<span class="count-k">${label}</span><span class="count-n">${counts[level] || 0}</span>`;
      summary.appendChild(cell);
    });

  const rows = el("watch-rows");
  rows.innerHTML = "";
  const districts = payload.districts || [];
  if (!districts.length) {
    rows.innerHTML = '<tr class="empty-row"><td colspan="5">Nothing crosses the alert ' +
      "threshold on this day. That is an all-clear, not a missing result.</td></tr>";
  }

  districts.forEach((district) => {
    const heavy = (district.heavy_rain_probability || {}).heavy || 0;
    const level = district.warning_level || "none";
    const tr = document.createElement("tr");
    tr.style.setProperty("--sev", `var(--sev-${level})`);
    tr.innerHTML =
      `<td><span class="rr-name">${district.district}` +
        `<span class="rr-type">${plainRegimeLabel(district.regime_label)}</span></span></td>` +
      `<td class="num">${fmt(district.corrected_forecast_mm, 0)}<i>mm</i></td>` +
      `<td class="num peak">${fmt(district.peak_cell_mm, 0)}<i>mm</i></td>` +
      `<td class="num">${pct(heavy)}</td>` +
      `<td class="right"><span class="tag ${
        level === "none" ? "tag-off" : level === "severe" ? "tag-bad" : "tag-warn"
      }">${(WARNING_TEXT[level] || WARNING_TEXT.none).headline}</span></td>`;
    tr.addEventListener("click", () => {
      el("location").value = district.district;
      showView("overview");
      run();
    });
    rows.appendChild(tr);
  });

  el("watch-status").textContent =
    `${payload.n_screened} districts screened.` + (payload.quiet ? " None reached the threshold." : "");

  const flagged = districts.filter((d) => d.warning_level !== "none").length;
  [el("watch-count"), el("alerts-pip")].forEach((badge) => {
    badge.textContent = flagged ? String(flagged) : "";
    badge.dataset.count = String(flagged);
  });
}

async function loadWatch(query) {
  const rows = el("watch-rows");
  rows.innerHTML = '<tr class="empty-row"><td colspan="5">Screening every district…</td></tr>';
  try {
    const payload = await getJSON(`/watch?date=${encodeURIComponent(query.date)}&limit=25`);
    renderWatch(payload);
    loaded.watch = query.date;
    updateHowtoForWatch(payload);
  } catch (err) {
    rows.innerHTML = `<tr class="empty-row"><td colspan="5">${err.message}</td></tr>`;
  }
}

/* --------------------------------------------------------------- events */

async function loadEvents() {
  const container = el("events");
  container.innerHTML = '<p class="row-note">Finding the biggest rainfall days…</p>';
  const unseenOnly = el("unseen-only").checked;

  let payload;
  try {
    payload = await getJSON(`/events?limit=8&unseen_only=${unseenOnly}`);
  } catch (err) {
    container.innerHTML = `<p class="row-note">${err.message}</p>`;
    return;
  }

  container.innerHTML = "";
  const events = payload.events || [];
  if (!events.length) {
    container.innerHTML = '<p class="row-note">No heavy-rain days found' +
      (unseenOnly ? " outside the training period. Untick the box to include days the system learned from." : ".") +
      "</p>";
  }

  events.forEach((event) => {
    const node = document.createElement("button");
    node.className = "event";
    const better = event.improved === true;
    node.innerHTML =
      `<span class="event-date">${event.date}</span>` +
      `<span class="event-name">${event.district}<small>${
        event.regime_label ? plainRegimeLabel(event.regime_label) : "type unknown"}</small></span>` +
      `<span class="event-fig"><span class="k">Existing</span><span class="v">${
        fmt(event.raw_forecast_mm, 1)}</span></span>` +
      `<span class="event-fig"><span class="k">Ours</span><span class="v ours">${
        fmt(event.corrected_forecast_mm, 1)}</span></span>` +
      `<span class="event-fig"><span class="k">Measured</span><span class="v truth">${
        fmt(event.observed_mm, 1)}</span></span>` +
      (event.improved === null || event.improved === undefined ? ""
        : `<span class="tag ${better ? "tag-ok" : "tag-bad"}">${better ? "Improved" : "No gain"}</span>`) +
      // Days the system trained on prove nothing, and must say so on the row.
      (event.in_training_period ? '<span class="tag tag-off">Trained on</span>' : "");
    node.addEventListener("click", () => {
      el("location").value = event.district;
      el("date").value = event.date;
      showView("overview");
      run();
    });
    container.appendChild(node);
  });

  el("events-status").textContent =
    `${payload.n_candidates} heavy district-days in the record.` +
    (payload.test_period_start
      ? ` Days from ${payload.test_period_start} onward were never used for training.`
      : "");
  loaded.replay = "loaded";
  updateHowtoForEvents(payload);
}

/* ------------------------------------------------------------- scenario */

const SCENARIO_CONTROLS = [
  { key: "humidity", label: "Moisture in the air", min: -30, max: 30, step: 1, unit: "%", hint: "More moisture usually means more rain" },
  { key: "wind", label: "Monsoon wind strength", min: -15, max: 15, step: 0.5, unit: " m/s", hint: "Stronger flow carries more moisture inland" },
  { key: "pressure", label: "Air pressure", min: -20, max: 20, step: 0.5, unit: " hPa", hint: "Lower pressure suggests a storm system" },
  { key: "instability", label: "Atmospheric instability", min: -2000, max: 2000, step: 50, unit: " J/kg", hint: "More instability means stronger updraughts" },
];

function renderScenarioControls() {
  const container = el("scenario-sliders");
  if (container.dataset.built === "true") return;

  SCENARIO_CONTROLS.forEach((control) => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML =
      `<span class="slider-head"><label for="sc-${control.key}">${control.label}</label>` +
      `<output id="sc-${control.key}-value">0${control.unit}</output></span>` +
      `<input id="sc-${control.key}" type="range" min="${control.min}" max="${control.max}" ` +
      `step="${control.step}" value="0">` +
      `<span class="unit">${control.hint}</span>`;
    container.appendChild(row);

    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      el(`sc-${control.key}-value`).textContent =
        `${input.value > 0 ? "+" : ""}${input.value}${control.unit}`;
    });
    // Only fire the request when the user lets go, not on every pixel.
    input.addEventListener("change", runScenario);
  });

  container.dataset.built = "true";
  runScenario();
}

async function runScenario() {
  if (state.staticMode) {
    el("scenario-disclaimer").textContent =
      "The scenario probe re-runs the model for every adjustment, so it needs " +
      "the live service. Run the project locally to use this panel.";
    return;
  }
  const query = currentQuery();
  if (!query.district || !query.date) {
    el("scenario-disclaimer").textContent = "Choose a district and date in the parameters panel first.";
    return;
  }
  const deltas = SCENARIO_CONTROLS
    .map((c) => `${c.key}=${encodeURIComponent(el(`sc-${c.key}`).value)}`).join("&");

  let payload;
  try {
    payload = await getJSON(
      `/what-if?date=${encodeURIComponent(query.date)}` +
      `&district=${encodeURIComponent(query.district)}&${deltas}`
    );
  } catch (err) {
    el("scenario-disclaimer").textContent = err.message;
    return;
  }

  el("scenario-before").textContent = mm(payload.baseline.corrected_forecast_mm);
  el("scenario-after").textContent = mm(payload.scenario.corrected_forecast_mm);
  el("scenario-before-regime").textContent = plainRegimeLabel(payload.baseline.regime_label);
  el("scenario-after-regime").innerHTML = plainRegimeLabel(payload.scenario.regime_label) +
    (payload.regime_changed ? ' <b style="color:var(--brass)">— type changed</b>' : "");

  const chips = el("scenario-deltas");
  chips.innerHTML = "";
  const rainDelta = payload.delta_corrected_mm;
  const addChip = (label, value, raise) => {
    const chip = document.createElement("span");
    chip.className = "d";
    chip.innerHTML = `${label} <b style="color:${raise ? "var(--cyan)" : "var(--coral)"}">${value}</b>`;
    chips.appendChild(chip);
  };
  addChip("Rainfall", `${rainDelta >= 0 ? "+" : "−"}${mm(Math.abs(rainDelta))}`, rainDelta >= 0);
  Object.entries(payload.delta_probability || {}).forEach(([name, delta]) => {
    if (Math.abs(delta) < 0.005) return;
    addChip(THRESHOLD_LABEL[name] || name,
      `${delta >= 0 ? "+" : "−"}${Math.round(Math.abs(delta) * 100)} pts`, delta >= 0);
  });
  if (chips.children.length === 1 && Math.abs(rainDelta) < 0.05) addChip("Risk", "unchanged", false);

  const ignored = (payload.adjustments || []).filter((a) => !a.applied);
  el("scenario-disclaimer").textContent = payload.disclaimer +
    (ignored.length
      ? ` Note: ${ignored.map((a) => a.control).join(", ")} had no effect because the connected ` +
        "dataset does not carry those measurements."
      : "");
}

/* --------------------------------------------------- "explain this page"
 *
 * Two layers. Static text says what a panel is for. The `live` lines are
 * computed from the numbers currently on screen, so the guidance reacts to the
 * actual forecast -- a split weather type, an average that disagrees with the
 * wettest cell. Those are the things a reader misreads.
 */

const HOWTO_STORAGE_KEY = "rainfall-howto-mode";
const THEME_STORAGE_KEY = "rainfall-theme";

const HOWTO = {
  "howto-intro": {
    title: "What you are looking at",
    body:
      "<p>Weather models are systematically wrong in ways that depend on the " +
      "<strong>kind of weather</strong> — they under-predict rain over hills, " +
      "over-predict it during dry spells. This system learns those patterns and " +
      "corrects them.</p>" +
      "<p>The page reads left to right: <strong>what you asked for</strong>, " +
      "<strong>what we forecast</strong>, <strong>why</strong>, and then " +
      "<strong>whether the system can be trusted at all</strong>.</p>",
  },
  "howto-rainfall": {
    title: "About these numbers",
    body:
      "<p>All four are averages across the whole district. &ldquo;Measured&rdquo; " +
      "only exists for days in the past — for a real forecast there is nothing to " +
      "compare against yet.</p>",
    watch:
      "One day tells you almost nothing. A correction can help on Monday and " +
      "hurt on Tuesday. Model Comparison is the section that counts.",
  },
  "howto-regime": {
    title: "Why weather type matters",
    body:
      "<p>Rain over hills behaves nothing like rain in a storm system, and a " +
      "weather model gets each of them wrong in a different direction. This system " +
      "keeps a separate specialist for each, which is the whole idea.</p>",
    watch:
      "Two types at once is not a failure — a coastal hill range genuinely is both, " +
      "and the system blends the specialists rather than forcing a choice.",
  },
  "howto-probability": {
    title: "How to read the odds",
    body:
      "<p>These are the chances that rainfall crosses each official danger " +
      "threshold. They are worked out separately from the amount above, so a " +
      "moderate forecast can still carry a real risk of something worse.</p>",
    watch:
      "A percentage is only worth anything if it is honest. Model Comparison " +
      "checks exactly that — read it before relying on these.",
  },
  "howto-warning": {
    title: "Where this comes from",
    body:
      "<p>The advisory level is set by fixed cut-offs on the percentages beside it: " +
      "30% triggers a watch, 50% a warning.</p>",
    watch:
      "Those cut-offs are provisional starting values, not thresholds agreed with " +
      "any weather agency. They would need setting properly before real use.",
  },
  "howto-maps": {
    title: "How to read the fields",
    body:
      "<p>Every panel shows the same day and the same area on the same colour scale, " +
      "so differences between them are real. The last one is the interesting one: " +
      "it shows only what the system <em>changed</em>.</p>",
    watch:
      "If the changes are scattered randomly, the system is just adding noise. If " +
      "they form patches — a band along the coast, a patch over the hills — it has " +
      "found a real, repeatable flaw in the weather model. That is the whole claim, " +
      "in one picture.",
  },
  "howto-bias": {
    title: "How to read these bars",
    body:
      "<p>Each bar is one piece of evidence and how far it pushed the forecast, " +
      "measured in millimetres. Bars to the right added rain, bars to the left " +
      "removed it. Together they add up to the change that was made.</p>",
    watch:
      "These explain the single wettest grid cell in the district, not the district " +
      "average in the cards above — which is why the numbers are bigger.",
  },
  "howto-regime-shap": {
    title: "A different question",
    body:
      "<p>These bars explain why the system decided <em>what kind of weather</em> " +
      "this is, rather than how much rain to expect. Same idea, different question " +
      "— which is why they are not in millimetres.</p>",
  },
  "howto-timeline": {
    title: "What this plots",
    body:
      "<p>The same district across several days. Each point is a whole day because " +
      "that is the shortest period the dataset covers &mdash; an hourly curve over " +
      "daily data would be invented detail.</p>",
    watch:
      "A 'type shift' tag means the weather type changed from the day before. That " +
      "is the moment the system swaps which specialist model it uses, so it is also " +
      "the moment the correction can change character.",
  },
  "howto-drivers": {
    title: "What this shows",
    body:
      "<p>Which conditions are moving the corrections most across the whole area " +
      "today, averaged over the grid rather than a single district.</p>",
    watch:
      "'Both ways' means a factor is strong everywhere but pushes rainfall up in " +
      "some places and down in others &mdash; usually terrain or coastline, which " +
      "cut differently depending on where you stand.",
  },
  "howto-watch": {
    title: "How to read this matrix",
    body:
      "<p>Every district scored for the selected day, worst first. " +
      "<strong>District average</strong> is the whole area; <strong>worst cell</strong> " +
      "is the single wettest grid square in it.</p>",
    watch:
      "Judge risk on the worst cell, not the average. A district can average a " +
      "harmless 15 mm while one valley inside it takes 90 mm, and it is the valley " +
      "that floods.",
  },
  "howto-replay": {
    title: "Why this matters",
    body:
      "<p>Anyone can look good on quiet days. These are the heaviest rainfall days " +
      "in the record &mdash; the ones that actually matter &mdash; with the whole " +
      "chain shown: what the weather model said, what we corrected it to, and what " +
      "fell.</p>",
    watch:
      "Days the system trained on are labelled. Getting those right proves nothing; " +
      "only days it had never seen are evidence.",
  },
  "howto-whatif": {
    title: "What this is for",
    body:
      "<p>A probe of the model, not a forecast. Moving a slider asks &lsquo;what " +
      "would the system have said if the air had been wetter?&rsquo; &mdash; useful " +
      "for checking that it responds to physics the way a forecaster would expect.</p>",
    watch:
      "Nothing here predicts the future. If a scenario produces an alarming number, " +
      "that is a statement about the model, not about the weather.",
  },
};

function renderHowtoShell() {
  Object.entries(HOWTO).forEach(([id, entry]) => {
    const node = el(id);
    if (!node) return;
    node.innerHTML =
      `<span class="howto-title">${entry.title}</span>` + entry.body +
      (entry.watch ? `<span class="watch"><strong>Watch out:</strong> ${entry.watch}</span>` : "");
  });
}

function addLiveNote(id, html, flag = false) {
  const node = el(id);
  if (!node) return;
  const existing = node.querySelector(".live");
  if (existing) existing.remove();
  const live = document.createElement("span");
  live.className = flag ? "live flag" : "live";
  live.innerHTML = `<strong>On this run:</strong> ${html}`;
  node.appendChild(live);
}

function updateHowtoForPrediction(data) {
  const blend = data.regime_blend || [];
  const leader = blend[0];
  if (leader && blend.length === 1) {
    addLiveNote("howto-regime",
      `the system is ${pct(leader.probability)} sure this is ` +
      `${regimeInfo(leader.regime).short.toLowerCase()}, so a single specialist handled the correction.`);
  } else if (leader) {
    addLiveNote("howto-regime",
      `this is a genuine mixture — ${regimeInfo(leader.regime).short.toLowerCase()} ` +
      `${pct(leader.probability)} and ${regimeInfo(blend[1].regime).short.toLowerCase()} ` +
      `${pct(blend[1].probability)}. Treat it as less certain than a clean call.`, true);
  }

  const probs = Object.entries(data.heavy_rain_probability || {});
  const live = probs.filter(([, v]) => v >= 0.01);
  if (probs.length && !live.length) {
    addLiveNote("howto-probability",
      "every risk is under 1%. Either this really is a quiet day, or the rarer " +
      "thresholds had too few past events to learn from.", true);
  } else if (live.length) {
    addLiveNote("howto-probability",
      `the strongest signal is ${(THRESHOLD_LABEL[live[0][0]] || live[0][0]).toLowerCase()} ` +
      `at ${pct(live[0][1])}.`);
  }

  const detail = data.bias_explanation;
  if (detail && detail.regime === "__fallback__") {
    addLiveNote("howto-bias",
      "no specialist existed for this weather type — there were too few past examples " +
      "to train one — so a general-purpose model made this correction.", true);
  } else if (detail) {
    addLiveNote("howto-bias",
      `the ${regimeInfo(detail.regime).short.toLowerCase()} specialist made this correction, ` +
      `changing the wettest cell by ${mm(Math.abs(detail.predicted_bias_mm))}.`);
  }

  if (data.raw_forecast_mm !== null && data.raw_forecast_mm !== undefined
      && data.observed_mm !== null && data.observed_mm !== undefined) {
    const before = Math.abs(data.raw_forecast_mm - data.observed_mm);
    const after = Math.abs(data.corrected_forecast_mm - data.observed_mm);
    addLiveNote("howto-rainfall",
      after < before
        ? `the correction moved the forecast closer to what fell, from ${mm(before)} off to ${mm(after)} off.`
        : `the correction moved the forecast further from what fell, from ${mm(before)} off to ` +
          `${mm(after)} off. That happens; judge the system on the season, not the day.`,
      after >= before);
  }
}

function updateHowtoForReport(report) {
  const value = (id, key) => {
    const m = report.models[id];
    return m ? metricsFor(m)[key] : null;
  };
  // The one finding from the scorecard that changes how the maps should be
  // read: when the no-AI ablation places rain better than we do, say so.
  const ourFss = value(OUR_MODEL, "fss");
  const ablationFss = value("C_regime_quantile_mapping", "fss");
  if (Number.isFinite(ourFss) && Number.isFinite(ablationFss) && ablationFss > ourFss) {
    addLiveNote("howto-maps",
      `on placing rain correctly, the simple weather-type-aware method is currently beating ` +
      `the AI one (${fmt(ablationFss, 2)} against ${fmt(ourFss, 2)}). Knowing the weather type ` +
      `is doing more of the work than the AI is.`, true);
  }
}

function updateHowtoForWatch(payload) {
  const flagged = (payload.districts || []).filter((d) => d.warning_level !== "none");
  if (!flagged.length) {
    addLiveNote("howto-watch",
      `all ${payload.n_screened} districts are below the alert threshold today. ` +
      `That is a genuine all-clear, not a failed lookup.`);
    return;
  }
  // Where the district average understates the peak, say so with the numbers.
  const worst = flagged[0];
  const spread = (worst.peak_cell_mm || 0) - (worst.corrected_forecast_mm || 0);
  addLiveNote("howto-watch",
    `${flagged.length} of ${payload.n_screened} districts need a look. ` +
    (spread > 20
      ? `${worst.district} averages only ${mm(worst.corrected_forecast_mm)} but its worst cell ` +
        `reaches ${mm(worst.peak_cell_mm)} — the average would have hidden that.`
      : `${worst.district} is the most exposed.`),
    spread > 20);
}

function updateHowtoForEvents(payload) {
  const events = payload.events || [];
  const scored = events.filter((e) => e.improved !== null && e.improved !== undefined);
  if (!scored.length) return;
  const better = scored.filter((e) => e.improved).length;
  addLiveNote("howto-replay",
    `the correction improved on the weather model in <strong>${better} of ${scored.length}</strong> ` +
    `of these events. ` +
    (better === scored.length
      ? "Every one is a small sample; Model Comparison tests a whole season."
      : "It is not meant to win every day — Model Comparison is the real test."),
    better < scored.length / 2);
}

function setHowtoMode(on) {
  document.body.classList.toggle("howto-on", on);
  el("howto-toggle").setAttribute("aria-pressed", String(on));
  try {
    localStorage.setItem(HOWTO_STORAGE_KEY, on ? "on" : "off");
  } catch (err) {
    /* Private browsing and blocked site data both throw; the mode still works
     * for this visit, it just will not be remembered. */
  }
}

function initHowto() {
  renderHowtoShell();
  let stored = null;
  try { stored = localStorage.getItem(HOWTO_STORAGE_KEY); } catch (err) { stored = null; }
  setHowtoMode(stored === "on");
  el("howto-toggle").addEventListener("click", () => {
    setHowtoMode(!document.body.classList.contains("howto-on"));
  });
}

/* ------------------------------------------------------------------ theme */

/* Dark is the intended look; the choice is remembered per browser once made. */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const button = el("theme-toggle");
  if (button) button.title = theme === "light" ? "Switch to dark" : "Switch to light";
  // Canvases bake their colours into pixels, so they have to be drawn again.
  if (state.lastTimeline) renderTrendChart(state.lastTimeline);
  if (state.lastGrid) renderMaps(state.lastGrid);
}

function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem(THEME_STORAGE_KEY); } catch (err) { stored = null; }
  applyTheme(stored === "light" ? "light" : "dark");
  el("theme-toggle").addEventListener("click", () => {
    const next = isLight() ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (err) { /* not fatal */ }
  });
}

/* ------------------------------------------------------------------- main */

async function run() {
  const query = currentQuery();
  if (!query.district || !query.date) {
    showAlert("Choose a district and a date in the parameters panel first.", "info");
    return;
  }

  const button = el("run");
  const label = button.querySelector("span");
  button.disabled = true;
  label.textContent = "Running…";
  clearAlert();

  try {
    const base = `date=${encodeURIComponent(query.date)}&soft_routing=${query.soft}`;
    const withThreshold = query.threshold ? `${base}&threshold=${query.threshold}` : base;
    const prediction = await getJSON(
      `/predict?${withThreshold}&district=${encodeURIComponent(query.district)}`
    );
    state.lastPrediction = prediction;

    renderHero(prediction, query.date);
    renderRegime(prediction);
    renderProbabilities(prediction);
    renderWarning(prediction);
    renderBiasExplanation(prediction);
    renderShap("regime-shap", prediction.explanation && prediction.explanation.top_features, "");
    renderChain(prediction);
    updateHowtoForPrediction(prediction);

    // Maps, the trend window and the risk matrix are independent: one failing
    // must not take the others down, nor the forecast already on screen.
    getJSON(`/grid?${base}`)
      .then((grid) => { renderMaps(grid); renderLineage(); })
      .catch((err) => {
        el("map-grid").innerHTML = "";
        el("map-status").textContent = `Fields unavailable: ${err.message}`;
      });

    getJSON(
      `/timeline?${base}&district=${encodeURIComponent(query.district)}` +
      `&back=${query.span}&forward=${query.span}`
    )
      .then(renderTimeline)
      .catch((err) => {
        el("chart-empty").textContent = err.message;
        el("chart-empty").classList.remove("hidden");
        el("timeline").innerHTML = "";
      });

    getJSON(`/drivers?date=${encodeURIComponent(query.date)}`)
      .then(renderDrivers)
      .catch(() => { el("drivers").innerHTML = '<p class="row-note">Drivers unavailable.</p>'; });

    // The watchlist is date-scoped, so a new date invalidates it.
    if (loaded.watch !== query.date) { loaded.watch = null; loadWatch(query); }
    if (state.view === "scenario") runScenario();
  } catch (err) {
    showAlert(err.message);
  } finally {
    button.disabled = false;
    label.textContent = "Run Analysis";
  }
}

/* ----------------------------------------------------------------- wiring */

function syncThresholdReadout() {
  el("threshold-out").textContent = `Alert me above ${el("threshold").value} mm`;
}

el("run").addEventListener("click", run);
el("threshold").addEventListener("input", syncThresholdReadout);
el("show-observed").addEventListener("change", () => {
  if (state.lastGrid) renderMaps(state.lastGrid);
});
el("unseen-only").addEventListener("change", loadEvents);
el("timespan").addEventListener("change", () => { if (state.lastPrediction) run(); });
["location", "date", "district"].forEach((id) => {
  el(id).addEventListener("keydown", (event) => { if (event.key === "Enter") run(); });
});
// The header search is a shortcut into the same parameter.
el("district").addEventListener("change", () => {
  const value = el("district").value.trim();
  if (value) { el("location").value = value; run(); }
});
el("alerts-btn").addEventListener("click", () => showView("risk"));
document.querySelectorAll("#nav button").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});
el("view-raw").addEventListener("click", () => {
  showAlert(JSON.stringify(
    { health: state.health, dates: state.dates, districts: state.districtSource }, null, 2
  ), "info");
});

/* Forecast CSV for the whole domain on the selected day. Rows come from the
 * risk matrix, which is the only endpoint that already scores every district. */
el("export-csv").addEventListener("click", async () => {
  const query = currentQuery();
  if (!query.date) {
    showAlert("Choose a date first — there is nothing to export yet.", "info");
    return;
  }
  let payload;
  try {
    payload = await getJSON(`/risk-matrix?date=${encodeURIComponent(query.date)}`);
  } catch (err) {
    showAlert(err.message);
    return;
  }
  const rows = payload.districts || [];
  if (!rows.length) {
    showAlert("No districts scored for that date, so the export would be empty.", "info");
    return;
  }

  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const t = String(v);
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
  };
  const header = [
    "District", "Latitude", "Longitude", "Valid Time",
    "Existing Forecast (mm)", "AI Forecast (mm)", "Measured (mm)",
    "Peak Cell (mm)", "Rainfall Category", "Advisory",
  ];
  const body = rows.map((d) => [
    d.district,
    d.centroid_lat === null || d.centroid_lat === undefined ? "" : fmt(d.centroid_lat, 3),
    d.centroid_lon === null || d.centroid_lon === undefined ? "" : fmt(d.centroid_lon, 3),
    payload.date,
    d.raw_forecast_mm === null || d.raw_forecast_mm === undefined ? "" : fmt(d.raw_forecast_mm, 2),
    fmt(d.corrected_forecast_mm, 2),
    d.observed_mm === null || d.observed_mm === undefined ? "" : fmt(d.observed_mm, 2),
    d.peak_cell_mm === null || d.peak_cell_mm === undefined ? "" : fmt(d.peak_cell_mm, 2),
    (categoryFor(d.corrected_forecast_mm) || {}).label || "",
    (WARNING_TEXT[d.warning_level] || WARNING_TEXT.none).headline,
  ].map(esc).join(","));

  const blob = new Blob([header.join(",") + "\n" + body.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `forecast_${payload.date}.csv`;
  link.click();
  URL.revokeObjectURL(url);
});

el("export-report").addEventListener("click", async () => {
  try {
    const payload = await getJSON("/verification-report");
    if (!payload.available) {
      showAlert(payload.detail || "No report has been generated yet.", "info");
      return;
    }
    const blob = new Blob([JSON.stringify(payload.report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "verification_report.json";
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showAlert(err.message);
  }
});

// The chart is drawn at device resolution, so it must be redrawn when the
// element's size changes rather than being stretched by the browser.
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.lastTimeline) renderTrendChart(state.lastTimeline);
  }, 150);
});


document.querySelectorAll("#layer-toggle button").forEach((button) => {
  button.addEventListener("click", () => {
    state.mapLayer = button.dataset.layer;
    document.querySelectorAll("#layer-toggle button").forEach((b) => {
      b.classList.toggle("on", b === button);
    });
    if (state.lastGrid) renderHeroMap(state.lastGrid);
  });
});

(function wireMapHover() {
  const canvas = el("map-hero");
  const tip = el("map-tip");
  if (!canvas || !tip) return;
  const show = (event) => {
    if (!state.lastGrid) return;
    const hit = cellAt(state.lastGrid, canvas, event.clientX, event.clientY);
    tip.innerHTML = describeCell(state.lastGrid, hit);
    tip.classList.toggle("on", !!hit);
  };
  canvas.addEventListener("mousemove", show);
  canvas.addEventListener("mouseleave", () => tip.classList.remove("on"));
  // Touch gets the same readout on tap.
  canvas.addEventListener("touchstart", (event) => {
    if (event.touches.length) show(event.touches[0]);
  }, { passive: true });
})();

syncThresholdReadout();
initTheme();
initHowto();
loadService();
