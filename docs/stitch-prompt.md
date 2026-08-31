# Stitch prompts — Forecasting Lab

Stitch works best iteratively: paste **Prompt 1** to establish the product and art
direction, then paste each screen prompt one at a time as follow-ups in the same
session so it keeps the design system consistent.

Everything below is grounded in what the backend actually serves. The sample
numbers are real values from the pipeline, so the generated screens will look
plausible rather than inventing meteorology.

---

## Prompt 1 — Product + art direction (paste this first)

```
Design a sophisticated web dashboard for "Forecasting Lab", a scientific
instrument that corrects monsoon rainfall forecasts for India using AI.

WHAT IT DOES
A weather model produces a rainfall forecast. It is systematically wrong in
patterns that depend on what kind of weather is happening — too dry over hills,
too wet during a lull. Our system classifies the weather type first, then applies
a correction specialist trained for that situation. The dashboard's whole job is
to show three numbers side by side and let a meteorologist judge whether the
correction helped:
  1. EXISTING FORECAST — what the weather model already predicted
  2. OUR PREDICTION — what our AI corrected it to
  3. ACTUAL RAINFALL — what was really measured (only exists for past dates)

AUDIENCE
Government meteorologists and disaster-management officers. They are experts, but
the interface must stay readable under pressure. Data density is expected;
decoration is not.

ART DIRECTION — "instrument-grade editorial"
Not a typical SaaS dashboard. Think Kinfolk or a Leica manual rather than a
crypto app. Restrained, confident, lots of negative space around very large
numbers.

- Ground: deep ink navy #0B1016, panels #141B24, raised surfaces #1C2530
- Type: warm ivory #F2EFE9 primary, muted #9AA5B1 secondary
- Accents, used sparingly and always meaning something:
    soft cyan #7FD1DE  = our corrected prediction
    dusty indigo #8B93C9 = the existing forecast
    warm brass  #C9A227 = actual measured rainfall
    muted coral #D4736A = severe advisory
- Typography: a high-contrast serif display face (Fraunces or Playfair Display)
  ONLY for the three big rainfall numbers, which should be enormous — 72px+.
  Everything else in a clean grotesque (Inter or Söhne). Small labels in
  uppercase at 11px with generous 0.14em letter-spacing.
- Geometry: 4px corner radius, hairline 1px borders at 8% white, no heavy
  shadows, no glassmorphism, no neon glow, no gradient meshes.
- Generous padding: 32px inside panels, 24px between them.

LAYOUT SHELL (same on every screen)
- Fixed 280px left sidebar: wordmark "FORECASTING LAB" with a small serif
  monogram, then nav — Overview, Historical Data, Risk Matrix, Scenario Lab,
  Settings. Active item marked with a thin brass rule on the left, not a filled
  pill. At the bottom: an "Export report" ghost button and a small live status
  dot reading "System ready".
- Top bar: screen title in the serif face, a district search field, a
  dark/light theme toggle, and a notification bell with a count badge.
- Support BOTH a dark and a light theme. Light theme is warm paper #F7F5F1 with
  ink #10161D, not clinical white.

CRITICAL CONSTRAINT
The dataset is DAILY (24-hour rainfall accumulation). Never design an hourly
chart, a live ticker, a "last updated 2 minutes ago" stamp, or anything implying
real-time streaming. Time axes are in days.
```

---

## Prompt 2 — Overview screen (the main one)

```
Now design the Overview screen for Forecasting Lab.

LEFT COLUMN (320px, sticky):
- "Simulation Parameters" panel: a district field (value "Mumbai, Maharashtra"),
  a date field (8 September 2022), a "Time span" dropdown set to "5 days", and a
  rainfall threshold slider labelled "Alert me above" set to 64.5 mm. Below them
  a full-width primary button "Run analysis".
- "Prediction Confidence" panel: a large thin-stroke circular gauge reading 99%,
  with "Low-pressure system" beneath it and the caption "A storm system is
  sitting overhead — heavy, concentrated rain." Under that, a small ranked list
  of five weather types with thin probability bars:
  Low-pressure system 99%, Active monsoon 1%, Monsoon break <1%,
  Hill rain <1%, Coastal rain <1%.

MAIN COLUMN:

1. FORECAST COMPARISON — the hero. One wide panel containing three figures in a
   row, connected left to right by thin arrows. Numbers in the serif display
   face at 72px+:
     Existing forecast — 158 mm — label "Already in your database"
     [arrow annotated "−37.2 mm"]
     Our prediction — 121 mm — label "81.5–164.6 · 80% confidence band"
     [arrow annotated "12.8 mm off"]
     Actual rainfall — 108 mm — label "we closed 37.2 mm of the gap"
   Each number tinted to its accent colour (indigo / cyan / brass).
   A single quiet line underneath: "A normal day here this month brings about
   5.9 mm. This is 114.6 mm more than usual."

2. RAINFALL TREND — a wide line chart, 5 daily points (6–10 September). Three
   smooth spline series matching the accent colours: our prediction (cyan, with
   a soft gradient area fill beneath it — this is the one the eye should land
   on), existing forecast (indigo), actual rainfall (brass, dashed). Very faint
   grid lines. Below the chart, a row of five small day chips showing 8.1, 18.2,
   121, 15.3, 16.0 mm — the middle one highlighted as "today", and two chips
   tagged "type shift".

3. Two panels side by side:
   - "Exceedance probability" — four rows with thin horizontal bars:
     Heavy rain (over 64.5 mm) 83%, Very heavy rain (over 115.6 mm) 69%,
     Extreme rain (over 204.4 mm) 9%, Your limit (over 64.5 mm) 83%.
   - "Advisory status" — a compact panel reading "SEVERE WARNING" in muted coral
     with the line "Very heavy rain is likely. Treat this as a serious risk."

4. "Why the forecast changed" — two columns of diverging horizontal bars showing
   feature contributions in millimetres. Left column "Contribution to correction":
   The weather model's own forecast −38.1 mm, Low-level wind north–south +3.5 mm,
   Location north–south +3.2 mm, Forecast detail −2.8 mm, Onshore wind −2.4 mm.
   Right column "Why this weather type": four rows labelled supports / against.
   Bars extend left and right from a centre line.

5. "Gridded field output" — three small square heatmaps side by side over a
   coarse grid, labelled "Weather model", "Our correction", "Change applied".
   The third uses a diverging blue–red scale. A slim colour legend underneath.

6. "Data sources" — a quiet table, visually lighter than the panels above:
   columns Source, Dataset, Resolution, Update, Status. Rows for the analysis
   table, district boundaries, weather-type classifier, correction models,
   probability models, and verification report — each with a small status pill.
```

---

## Prompt 3 — Risk Matrix screen

```
Design the "Risk Matrix" screen for Forecasting Lab.

A summary strip of four counts across the top: Severe 2, Warning 8, Watch 12,
Clear 27 — each a large numeral with a small uppercase label, tinted by severity.

Below it, a ranked list of districts for the selected day, worst first. Each row
shows: district name with its weather type underneath in small muted text, then
three aligned numeric columns — "District average" 121 mm, "Worst cell" 210 mm,
"Heavy rain" 83% — and a severity pill on the right (SEVERE / WARNING / WATCH).
A thin severity-coloured rule runs down the left edge of each row.

Include a short note above the list: "Judge risk on the worst cell, not the
average. A district can average a harmless 15 mm while one valley inside it
takes 90 mm."
```

---

## Prompt 4 — Historical Data screen

```
Design the "Historical Data" screen for Forecasting Lab — an archive of the
heaviest past rainfall days, used to check the correction against days the model
never trained on.

A toggle at the top right: "Held-out days only".

A list of event rows. Each row: the date on the left, then the district with its
weather type beneath, then three aligned figures — Existing 157.7, Ours 120.5,
Measured 107.8 mm — then a small outcome pill reading "Improved" or "No gain".
Rows for days the model trained on carry a second grey pill reading "Trained on",
because getting those right proves nothing.
```

---

## Prompt 5 — Scenario Lab screen

```
Design the "Scenario Lab" screen for Forecasting Lab — a probe that re-runs the
model under altered atmospheric conditions.

Left half: four labelled sliders, each with a live value and a one-line hint —
Moisture in the air (+0%), Monsoon wind strength (+0 m/s), Air pressure (0 hPa),
Atmospheric instability (0 J/kg).

Right half: a before/after comparison — "Baseline 121 mm" and "Perturbed 121 mm"
side by side with an arrow between them, the weather type named under each. Below
that, a short list of deltas (rainfall, heavy rain odds).

At the bottom, a persistent warning notice in muted brass: "A model scenario, not
a forecast. This shows how the model responds to altered inputs; it says nothing
about what the atmosphere will do." This notice must always be visible.
```

---

## Things to tell Stitch NOT to do

Paste this if it drifts:

```
Corrections:
- No hourly or real-time data. The dataset is daily.
- No fake precision — do not invent decimals beyond one place.
- No emoji as icons; use a single consistent line-icon set.
- No glassmorphism, neon glow, gradient mesh backgrounds, or 3D illustrations.
- Do not use the word "regime" in the interface. Say "weather type".
- Keep body text at 4.5:1 contrast minimum in both themes.
```
