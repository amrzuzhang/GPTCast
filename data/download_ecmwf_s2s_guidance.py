#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from gptcast.data.guidance_ecmwf import (
    SUPPORTED_ECMWF_INIT_WEEKDAYS,
    default_guidance_root,
    derive_candidate_init_dates,
)
from gptcast.data.ecmwf_s2s_download import ensure_ecmwf_guidance


def _parse_dates(items: str | None) -> list[datetime]:
    if items is None or str(items).strip() == "":
        return []
    out = []
    for token in str(items).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(datetime.strptime(token, "%Y-%m-%d"))
    return out


def _parse_ints(items: str) -> list[int]:
    return [int(x.strip()) for x in str(items).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECMWF S2S guidance fields for GPTCast hybrid experiments.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "0.1" / "1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metadata-path", action="append", default=[])
    parser.add_argument("--dates", default=None, help="Comma-separated init dates YYYY-MM-DD. Overrides metadata-derived dates if set.")
    parser.add_argument("--variables", default="ecmwf_sm100", help="Comma-separated guidance keys. Default: ecmwf_sm100")
    parser.add_argument("--lead-days", default="7", help="Comma-separated lead days. Default: 7")
    parser.add_argument("--weekdays", default="0,3", help="Allowed init weekdays when deriving from metadata. Default: Monday/Thursday.")
    parser.add_argument("--min-year", type=int, default=1995)
    parser.add_argument("--area", default="42.0,105.0,20.0,125.0", help="north,west,south,east")
    parser.add_argument("--grid", default="1.5/1.5", help="Target output grid, e.g. 1.5/1.5")
    parser.add_argument("--max-init-dates", type=int, default=None, help="Optional cap on the number of init dates to download (sorted ascending).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_dir or default_guidance_root(args.base_dir)
    variables = [x.strip() for x in str(args.variables).split(",") if x.strip()]
    lead_days = _parse_ints(args.lead_days)
    weekdays = _parse_ints(args.weekdays)
    area = [float(x.strip()) for x in str(args.area).split(",") if x.strip()]

    if args.dates:
        init_dates = _parse_dates(args.dates)
    else:
        metadata_paths = [Path(p) for p in args.metadata_path]
        if not metadata_paths:
            raise ValueError("Provide either --dates or at least one --metadata-path")
        init_dates = derive_candidate_init_dates(
            [str(p) for p in metadata_paths],
            weekdays=weekdays,
            min_year=args.min_year,
        )

    if not init_dates:
        raise RuntimeError("No candidate ECMWF init dates found for download.")

    if args.max_init_dates is not None:
        init_dates = init_dates[: int(args.max_init_dates)]

    print(f"Downloading ECMWF guidance for {len(init_dates)} init dates into {out_root}")
    created = ensure_ecmwf_guidance(
        variables=variables,
        init_dates=init_dates,
        out_root=out_root,
        lead_days=lead_days,
        area=area,
        grid=args.grid,
        overwrite=bool(args.overwrite),
    )
    print(f"Created/updated {len(created)} files")


if __name__ == "__main__":
    main()
