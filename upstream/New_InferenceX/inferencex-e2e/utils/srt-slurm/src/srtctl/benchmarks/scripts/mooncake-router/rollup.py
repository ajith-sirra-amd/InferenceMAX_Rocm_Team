#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json from aiperf/mooncake-router results."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(log_dir: Path) -> None:
    """Generate benchmark-rollup.json from aiperf result files."""
    artifacts = log_dir / "artifacts"
    aiperf_files = list(artifacts.glob("*/profile_export_aiperf.json")) if artifacts.exists() else []

    if not aiperf_files:
        print("No aiperf results found", file=sys.stderr)
        return

    # Use most recent file
    latest = max(aiperf_files, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text())
    except json.JSONDecodeError as e:
        print(f"Failed to parse {latest}: {e}", file=sys.stderr)
        return

    rollup = {
        "benchmark_type": "mooncake-router",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "workload": data.get("workload"),
            "model": data.get("model"),
        },
        "data": data,
    }

    output_path = log_dir / "benchmark-rollup.json"
    output_path.write_text(json.dumps(rollup, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs")
    main(log_dir)
