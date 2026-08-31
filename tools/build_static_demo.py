"""Freeze the running API into static JSON so the dashboard can be hosted alone.

The dashboard is not a standalone app: every number on screen is fetched from
the FastAPI service at the same origin. A static host has no service, so this
script records the API's answers for **one date** and writes them next to the
dashboard. ``app.js`` falls back to these files when the live service cannot be
reached, which is what makes a static deployment show a working screen instead
of an error page.

What this is not:

* It is not a way to ship results. The snapshot inherits whatever dataset the
  service was serving, and the manifest records that -- if the source was the
  synthetic demo set, ``synthetic`` is true and the dashboard forces its
  "everything here is fabricated" banner on and refuses to let the reader
  dismiss it.
* It is not a substitute for the backend. The what-if simulator re-runs the
  model per request, so it cannot be frozen; in static mode the dashboard says
  so rather than showing a stale answer.

Usage::

    python -m uvicorn rainfall_pipeline.api.main:app --port 8000   # in one shell
    python tools/build_static_demo.py --date 2022-09-08            # in another
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_BASE = "http://127.0.0.1:8000"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "rainfall_pipeline" / "api" / "static" / "demo"


def slugify(value: str) -> str:
    """Filesystem- and URL-safe key for a district name."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unnamed"


def fetch(base: str, path: str) -> Any:
    """GET one endpoint and return the decoded JSON.

    Raises:
        RuntimeError: If the service is unreachable or answers with an error.
    """
    url = f"{base}{path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"{path} returned HTTP {err.code}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(
            f"Could not reach {url}. Start the service first:\n"
            f"  ./tools/serve_demo.sh"
        ) from err


def write(target: Path, payload: Any) -> int:
    """Write ``payload`` as compact JSON and return the byte count."""
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    target.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def build(base: str, date: str, span: int, out_dir: Path) -> Dict[str, Any]:
    """Record every endpoint the dashboard reads, for a single date.

    Args:
        base: Origin of the running service.
        date: The valid date to freeze.
        span: Days either side of ``date`` for the timeline window.
        out_dir: Directory to write the snapshot into.

    Returns:
        The manifest that was written.
    """
    if out_dir.exists():
        for stale in sorted(out_dir.rglob("*.json")):
            stale.unlink()

    total = 0
    files: Dict[str, str] = {}

    def record(key: str, path: str, relative: str) -> Any:
        nonlocal total
        payload = fetch(base, path)
        total += write(out_dir / relative, payload)
        files[key] = relative
        return payload

    # -- date-independent -----------------------------------------------------
    health = record("health", "/health", "health.json")
    if not health.get("models_loaded") or not health.get("data_connected"):
        raise RuntimeError(
            "The service is up but not trained, so the snapshot would be empty.\n"
            "Run ./tools/run_demo.sh (or connect real data) first."
        )

    dates = record("dates", "/dates", "dates.json")
    districts = record("districts", "/districts", "districts.json")
    record("verification-report", "/verification-report", "verification-report.json")

    # -- date-scoped ----------------------------------------------------------
    q = urllib.parse.quote(date)
    record("grid", f"/grid?date={q}&soft_routing=true", "grid.json")
    record("watch", f"/watch?date={q}&limit=25", "watch.json")
    record("risk-matrix", f"/risk-matrix?date={q}", "risk-matrix.json")
    record("drivers", f"/drivers?date={q}", "drivers.json")
    for flag in ("true", "false"):
        record(f"events:{flag}", f"/events?limit=8&unseen_only={flag}", f"events-{flag}.json")

    # -- per district ---------------------------------------------------------
    names: List[str] = districts.get("districts", [])
    for name in names:
        slug = slugify(name)
        dq = urllib.parse.quote(name)
        record(
            f"predict:{slug}",
            f"/predict?date={q}&district={dq}&soft_routing=true",
            f"predict/{slug}.json",
        )
        record(
            f"timeline:{slug}",
            f"/timeline?date={q}&district={dq}&back={span}&forward={span}",
            f"timeline/{slug}.json",
        )

    manifest = {
        "date": date,
        "span": span,
        "districts": names,
        # Carried through so the dashboard can force its fabricated-data banner
        # on and keep it there. A snapshot of fake data is still fake data.
        "synthetic": bool(dates.get("synthetic")),
        "files": files,
    }
    total += write(out_dir / "index.json", manifest)

    print(f"wrote {len(files) + 1} files, {total / 1024:.0f} KB → {out_dir}")
    if manifest["synthetic"]:
        print("NOTE: the source dataset was synthetic. The dashboard will say so and")
        print("      will not let the reader dismiss the notice.")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help="origin of the running service")
    parser.add_argument("--date", required=False, help="valid date to freeze (default: last available)")
    parser.add_argument("--span", type=int, default=2, help="days either side for the timeline")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args(argv)

    try:
        date = args.date
        if not date:
            dates = fetch(args.base, "/dates")
            date = dates.get("end")
            if not date:
                raise RuntimeError("The service reports no available dates.")
            print(f"no --date given; using the last available date: {date}")
        build(args.base, date, args.span, args.out)
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
