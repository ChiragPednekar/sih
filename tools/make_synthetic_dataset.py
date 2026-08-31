"""Generate a SYNTHETIC dataset for exercising the pipeline end to end.

============================================================================
THIS IS FAKE DATA. IT IS NOT ERA5, NOT IMD, NOT ANY REAL FORECAST MODEL.
Nothing produced by this script may be presented as a result. Its only job is
to prove the pipeline runs, trains, routes, calibrates, aggregates and
verifies correctly before real data is connected.
============================================================================

This file lives in ``tools/`` and NOT inside the ``rainfall_pipeline`` package,
deliberately: it is not part of the production system and nothing in the
pipeline imports it. It writes to ``sample_data/``, never to ``data_store/``,
so it cannot collide with the real files.

What it fabricates
------------------
A 1-degree grid over 8-28N, 70-88E across three monsoon seasons (JJAS 2020-22),
with:

* an intraseasonal active/break oscillation driving large-scale rainfall and
  the 850 hPa westerly jet;
* synthetic monsoon depressions seeded over the Bay of Bengal that track
  west-north-west, carrying high vorticity, low pressure and heavy rain;
* a Western Ghats orographic ridge and a coastline, so the terrain-driven
  regimes have something to key on;
* a raw "NWP forecast" derived from the truth by applying a **regime-dependent
  error structure** plus noise.

That last point is the actual test. A forecast whose error is pure noise cannot
be corrected by anything, so the run would tell us nothing. Here the injected
bias differs by mechanism -- the fake NWP badly under-forecasts orographic
rain, mildly under-forecasts coastal rain, over-forecasts around depressions,
and drizzles during breaks -- which is a caricature of how real convection- and
orography-related NWP errors behave. Whether the regime-specific models recover
that structure better than a global one is then an honest question the
verification report answers rather than something rigged in advance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- domain -----------------------------------------------------------------
LAT_MIN, LAT_MAX, LAT_STEP = 8.0, 28.0, 1.0
LON_MIN, LON_MAX, LON_STEP = 70.0, 88.0, 1.0
SEASONS = (2020, 2021, 2022)
SEASON_START, SEASON_END = "06-01", "09-30"

#: A crude west-coast coastline: longitude of the coast at each latitude.
#: South of ~22N the west coast runs roughly NNW; north of that it bends away.
WEST_COAST_LON = 73.0
#: A crude east-coast line.
EAST_COAST_LON = 85.0

RNG_SEED = 20260826


def build_grid() -> pd.DataFrame:
    """Build the static (lat, lon, elevation, coastal_distance) grid.

    Returns:
        One row per grid cell with terrain and coastal geometry.
    """
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, LAT_STEP)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, LON_STEP)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()

    # --- terrain ---------------------------------------------------------
    # Western Ghats: a narrow high ridge just inland of the west coast,
    # running from ~8N to ~21N.
    ghats = 1500.0 * np.exp(-(((lon - 74.5) / 1.1) ** 2)) * np.clip(
        (21.5 - lat) / 13.0, 0.0, 1.0
    ) * (lat < 21.5)
    # Deccan plateau: a broad rise across peninsular India.
    plateau = 450.0 * np.exp(-(((lat - 16.0) / 6.0) ** 2)) * np.exp(
        -(((lon - 77.5) / 5.0) ** 2)
    )
    # Himalayan foothills along the northern edge.
    foothills = 2200.0 * np.clip((lat - 26.5) / 2.0, 0.0, 1.0)
    # Eastern hills over the far northeast.
    eastern = 900.0 * np.exp(-(((lon - 86.5) / 2.0) ** 2)) * np.clip(
        (lat - 23.0) / 4.0, 0.0, 1.0
    )
    elevation = np.clip(ghats + plateau + foothills + eastern, 0.0, None)

    # --- coast -----------------------------------------------------------
    # Distance to the nearer of the two coastlines, in km. Peninsular
    # latitudes only; further north the domain is inland.
    deg_km = 111.0
    to_west = np.abs(lon - WEST_COAST_LON) * deg_km * np.cos(np.radians(lat))
    to_east = np.abs(lon - EAST_COAST_LON) * deg_km * np.cos(np.radians(lat))
    # South of 22N both coasts are near; north of that, treat as inland by
    # adding the distance up to the peninsula's neck.
    inland_penalty = np.clip(lat - 22.0, 0.0, None) * deg_km
    coastal_distance = np.minimum(to_west, to_east) + inland_penalty

    return pd.DataFrame(
        {
            "lat": np.round(lat, 2),
            "lon": np.round(lon, 2),
            "elevation": np.round(elevation, 1),
            "coastal_distance": np.round(coastal_distance, 1),
        }
    )


def season_dates() -> pd.DatetimeIndex:
    """Return every JJAS date across the configured seasons."""
    parts = [
        pd.date_range(f"{year}-{SEASON_START}", f"{year}-{SEASON_END}", freq="D")
        for year in SEASONS
    ]
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts]))


def seed_depressions(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Seed synthetic monsoon depressions tracking WNW from the Bay of Bengal.

    Args:
        dates: The full date index.
        rng: Random generator.

    Returns:
        One row per (date, active depression) with its centre and intensity.
    """
    rows = []
    for year in SEASONS:
        season = dates[dates.year == year]
        # Roughly 2 per month over a 4-month season.
        n_systems = rng.integers(6, 11)
        genesis_days = rng.choice(len(season) - 6, size=n_systems, replace=False)
        for start in sorted(genesis_days):
            lat0 = rng.uniform(17.0, 22.0)
            lon0 = rng.uniform(84.0, 88.0)
            lifetime = int(rng.integers(3, 7))
            peak = rng.uniform(0.7, 1.4)
            for age in range(lifetime):
                idx = start + age
                if idx >= len(season):
                    break
                # WNW propagation, ~2.5 deg lon/day west and ~0.6 deg north.
                envelope = np.sin(np.pi * (age + 0.5) / lifetime) * peak
                rows.append(
                    {
                        "date": season[idx],
                        "dep_lat": lat0 + 0.6 * age,
                        "dep_lon": lon0 - 2.5 * age,
                        "dep_intensity": envelope,
                    }
                )
    return pd.DataFrame(rows)


def build_daily_state(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Build the large-scale daily state: the active/break oscillation.

    Real monsoon intraseasonal variability is dominated by a 30-60 day
    northward-propagating oscillation. This reproduces a caricature of it with
    two superposed periods plus noise, which is enough to make the rule-based
    active/break labeller produce both classes.

    Args:
        dates: The full date index.
        rng: Random generator.

    Returns:
        One row per date with the oscillation index and the jet strength.
    """
    # Restart the phase each season so the years are not one continuous cycle.
    day_of_season = np.zeros(len(dates))
    for year in SEASONS:
        mask = dates.year == year
        day_of_season[mask] = np.arange(mask.sum())

    miso = (
        0.75 * np.sin(2 * np.pi * day_of_season / 41.0 + rng.uniform(0, 2 * np.pi))
        + 0.35 * np.sin(2 * np.pi * day_of_season / 17.0 + rng.uniform(0, 2 * np.pi))
        + rng.normal(0, 0.22, len(dates))
    )
    # Seasonal envelope: July-August are the wettest part of JJAS.
    seasonal = np.sin(np.pi * np.clip(day_of_season / 121.0, 0, 1)) ** 0.6

    return pd.DataFrame(
        {
            "date": dates,
            "miso": miso,
            "seasonal": seasonal,
            # The low-level westerly jet strengthens in active phases.
            "core_u850": 6.5 + 4.2 * miso + 1.5 * seasonal + rng.normal(0, 0.8, len(dates)),
        }
    )


def generate(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the full synthetic table in the pipeline's common schema.

    Args:
        rng: Random generator.

    Returns:
        One row per (date, lat, lon) with every schema column populated.
    """
    grid = build_grid()
    dates = season_dates()
    daily = build_daily_state(dates, rng)
    depressions = seed_depressions(dates, rng)

    n_cells = len(grid)
    frames = []

    dep_by_date = {d: g for d, g in depressions.groupby("date")}

    for _, day in daily.iterrows():
        date = day["date"]
        miso = float(day["miso"])
        seasonal = float(day["seasonal"])
        core_u = float(day["core_u850"])

        lat = grid["lat"].to_numpy()
        lon = grid["lon"].to_numpy()
        elev = grid["elevation"].to_numpy()
        coast = grid["coastal_distance"].to_numpy()

        # --- depression influence ----------------------------------------
        dep_rain = np.zeros(n_cells)
        dep_vort = np.zeros(n_cells)
        dep_pres = np.zeros(n_cells)
        for _, dep in dep_by_date.get(date, pd.DataFrame()).iterrows():
            dist2 = ((lat - dep["dep_lat"]) / 2.6) ** 2 + ((lon - dep["dep_lon"]) / 3.2) ** 2
            influence = np.exp(-dist2) * float(dep["dep_intensity"])
            dep_rain += 55.0 * influence
            dep_vort += 9.0e-5 * influence
            dep_pres += -7.5 * influence

        # --- wind field ---------------------------------------------------
        # Westerly jet peaks around 12-15N over the Arabian Sea and weakens north.
        jet = core_u * np.exp(-(((lat - 13.5) / 8.5) ** 2)) + rng.normal(0, 1.1, n_cells)
        u850 = jet + 2.0 * np.exp(-(((lat - 20.0) / 6.0) ** 2)) * miso
        v850 = 2.2 + 1.6 * miso + rng.normal(0, 1.3, n_cells)
        # Depressions add cyclonic circulation to the low-level flow.
        u850 = u850 - 4.0 * dep_vort / 9.0e-5
        u200 = -18.0 - 4.0 * np.exp(-(((lat - 15.0) / 9.0) ** 2)) + rng.normal(0, 2.5, n_cells)
        v200 = rng.normal(0.5, 2.0, n_cells)
        wind_speed_850 = np.hypot(u850, v850)

        # --- thermodynamics -------------------------------------------------
        humidity = np.clip(
            62.0
            + 16.0 * seasonal
            + 9.0 * miso
            + 14.0 * np.exp(-coast / 220.0)
            + 30.0 * dep_vort / 9.0e-5
            + rng.normal(0, 4.5, n_cells),
            25.0,
            100.0,
        )
        cape = np.clip(
            300.0
            + 16.0 * (humidity - 60.0)
            + 420.0 * seasonal
            + 260.0 * miso
            + 700.0 * dep_vort / 9.0e-5
            + rng.normal(0, 190.0, n_cells),
            0.0,
            4200.0,
        )
        pressure_msl = (
            1008.0
            - 4.5 * seasonal
            - 1.8 * miso
            - 0.0009 * elev
            + dep_pres
            + rng.normal(0, 1.0, n_cells)
        )
        vorticity = (
            1.0e-5 * (0.6 + 0.9 * miso)
            + dep_vort
            + rng.normal(0, 0.7e-5, n_cells)
        )
        olr = np.clip(
            255.0 - 26.0 * seasonal - 16.0 * miso - 55.0 * dep_vort / 9.0e-5
            + rng.normal(0, 9.0, n_cells),
            110.0,
            300.0,
        )

        # --- true rainfall ---------------------------------------------------
        # Upslope forcing: westerly flow hitting the Ghats. Strongest where the
        # terrain is high AND the low-level wind is strong.
        upslope = np.clip(u850, 0, None) * np.clip(elev, 0, None) / 900.0
        # Coastal convergence, decaying inland.
        coastal = np.clip(u850, 0, None) * np.exp(-coast / 90.0) * 2.2
        # Large-scale monsoon rainfall over the core zone.
        broad = (
            14.0
            * seasonal
            * np.exp(-(((lat - 20.0) / 7.5) ** 2))
            * np.exp(0.55 * miso)
        )

        intensity = broad + 1.35 * upslope + coastal + dep_rain
        # Rain occurrence: not every cell rains every day.
        p_rain = 1.0 / (1.0 + np.exp(-(0.05 * intensity + 0.045 * (humidity - 68.0))))
        wet = rng.random(n_cells) < p_rain
        # Gamma-distributed amounts: right-skewed, as rainfall is.
        shape = 0.85
        scale = np.clip(intensity, 0.6, None) / shape
        observed = np.where(wet, rng.gamma(shape, scale), 0.0)
        observed = np.round(np.clip(observed, 0.0, 700.0), 1)

        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "lat": lat,
                    "lon": lon,
                    "pressure_msl": np.round(pressure_msl, 2),
                    "wind_u_850": np.round(u850, 2),
                    "wind_v_850": np.round(v850, 2),
                    "wind_u_200": np.round(u200, 2),
                    "wind_v_200": np.round(v200, 2),
                    "olr": np.round(olr, 1),
                    "humidity": np.round(humidity, 1),
                    "cape": np.round(cape, 1),
                    "vorticity": vorticity,
                    "elevation": elev,
                    "coastal_distance": coast,
                    "observed_mm": observed,
                    "_wind_speed_850": wind_speed_850,
                    "_dep": dep_vort / 9.0e-5,
                }
            )
        )

    table = pd.concat(frames, ignore_index=True)
    table["raw_forecast_mm"] = fake_nwp_forecast(table, rng)
    return table.drop(columns=["_wind_speed_850", "_dep"])


def fake_nwp_forecast(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Degrade the truth into a fake "raw NWP forecast".

    The error structure is deliberately **mechanism-dependent**, because that is
    the thing a regime-aware corrector is supposed to exploit and a regime-blind
    one is not. A forecast whose error were pure noise would make the whole
    verification exercise vacuous.

    The caricature applied here:

    * **Orographic** (high terrain, strong flow) -- severe dry bias. Coarse
      models genuinely under-resolve the Ghats and miss the upslope maximum.
    * **Coastal** -- moderate dry bias, for the same reason at smaller scale.
    * **Depression** -- wet bias plus a spatial smear, standing in for a
      displaced or over-deepened system.
    * **Break** -- spurious drizzle, the classic light-rain over-forecast.
    * **Everything else** -- mild wet bias at low intensities, mild dry bias at
      high ones, i.e. the usual compression of the distribution's tails.

    Args:
        table: The generated truth table.
        rng: Random generator.

    Returns:
        The fake forecast, in mm.
    """
    observed = table["observed_mm"].to_numpy()
    elev = table["elevation"].to_numpy()
    coast = table["coastal_distance"].to_numpy()
    wind = table["_wind_speed_850"].to_numpy()
    dep = table["_dep"].to_numpy()
    n = len(table)

    forecast = observed.copy()

    # Baseline tail compression: the model damps extremes and inflates drizzle.
    forecast = np.where(observed > 40.0, observed * 0.82, forecast)
    forecast = np.where(observed < 8.0, observed * 1.35 + 1.2, forecast)

    orographic = (elev >= 600.0) & (wind >= 6.0)
    forecast = np.where(orographic, forecast * 0.52, forecast)

    coastal = (coast <= 75.0) & (wind >= 8.0) & ~orographic
    forecast = np.where(coastal, forecast * 0.72, forecast)

    depression = dep > 0.35
    forecast = np.where(depression, forecast * 1.28 + 6.0, forecast)

    # Spurious drizzle on genuinely dry cells during weak flow.
    drizzle = (observed < 1.0) & (wind < 6.0)
    forecast = np.where(drizzle, forecast + rng.gamma(1.1, 2.6, n), forecast)

    # Irreducible noise: without it, the bias would be perfectly learnable and
    # every corrector would look implausibly good.
    forecast = forecast * rng.lognormal(0.0, 0.42, n) + rng.normal(0.0, 1.6, n)
    return np.round(np.clip(forecast, 0.0, None), 1)


def build_districts(grid: pd.DataFrame, block: float = 3.0):
    """Build synthetic rectangular "districts" covering the grid.

    These are named ``Zone-A1``, ``Zone-B2`` and so on -- deliberately not real
    district names, so no output of this run can be mistaken for a real
    district product.

    Args:
        grid: The static grid, used only for its extent.
        block: Side length of each synthetic district, in degrees.

    Returns:
        A GeoDataFrame of square polygons with a ``district`` column.
    """
    import geopandas as gpd
    from shapely.geometry import box

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names, polys = [], []
    row = 0
    lat = LAT_MIN - LAT_STEP / 2
    while lat < LAT_MAX + LAT_STEP / 2:
        col = 0
        lon = LON_MIN - LON_STEP / 2
        while lon < LON_MAX + LON_STEP / 2:
            names.append(f"Zone-{letters[row]}{col + 1}")
            polys.append(box(lon, lat, lon + block, lat + block))
            lon += block
            col += 1
        lat += block
        row += 1
    return gpd.GeoDataFrame(
        {"district": pd.Series(names, dtype="string"), "geometry": polys},
        crs="EPSG:4326",
    )


def main(argv: list[str] | None = None) -> int:
    """Write the synthetic dataset to disk in the pipeline's expected layout.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out-dir",
        default="sample_data",
        help="Where to write the synthetic files (never data_store/).",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    if out.name == "data_store":
        raise SystemExit(
            "Refusing to write synthetic data into data_store/, which is "
            "reserved for your real files."
        )
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    table = generate(rng)

    era5_cols = [
        "date", "lat", "lon", "pressure_msl", "wind_u_850", "wind_v_850",
        "wind_u_200", "wind_v_200", "olr", "humidity", "cape", "vorticity",
        "elevation", "coastal_distance",
    ]
    table[era5_cols].to_parquet(out / "era5.parquet", index=False)
    table[["date", "lat", "lon", "observed_mm"]].to_parquet(
        out / "observed_rainfall.parquet", index=False
    )
    table[["date", "lat", "lon", "raw_forecast_mm"]].to_parquet(
        out / "raw_nwp_forecast.parquet", index=False
    )
    build_districts(build_grid()).to_file(out / "districts.geojson", driver="GeoJSON")

    # A machine-readable marker so every downstream consumer -- above all the
    # dashboard -- can tell it is showing fabricated data and say so on screen.
    # Path sniffing would break the moment someone renames the directory.
    (out / "SYNTHETIC.marker").write_text(
        "This directory holds fabricated data produced by "
        "tools/make_synthetic_dataset.py. Nothing derived from it is a result.\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(
        "# SYNTHETIC DATA -- NOT REAL\n\n"
        "Every file in this directory was fabricated by "
        "`tools/make_synthetic_dataset.py` to test that the pipeline runs.\n\n"
        "It is not ERA5, not IMD, and not any real forecast model. No number "
        "computed from it means anything about real monsoon forecasting, and "
        "none of it may be presented as a result.\n\n"
        "Your real files go in `data_store/`, not here.\n",
        encoding="utf-8",
    )

    print(f"Wrote synthetic dataset to {out}/")
    print(f"  rows          : {len(table):,}")
    print(f"  grid cells    : {table[['lat', 'lon']].drop_duplicates().shape[0]:,}")
    print(f"  dates         : {table['date'].nunique():,} "
          f"({table['date'].min().date()} to {table['date'].max().date()})")
    print(f"  observed mm   : mean {table['observed_mm'].mean():.2f}, "
          f"max {table['observed_mm'].max():.1f}")
    print(f"  forecast mm   : mean {table['raw_forecast_mm'].mean():.2f}, "
          f"max {table['raw_forecast_mm'].max():.1f}")
    for name, mm in (("heavy", 64.5), ("very heavy", 115.6), ("extremely heavy", 204.4)):
        n = int((table["observed_mm"] > mm).sum())
        print(f"  obs > {mm:6.1f} mm : {n:,} rows ({100 * n / len(table):.2f}%)  [{name}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
