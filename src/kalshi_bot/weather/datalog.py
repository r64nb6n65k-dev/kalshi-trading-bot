"""Append-only JSONL log of model snapshots, market quotes and orders.

One file per UTC day under ``<data_dir>/log/``. This record is what
``weather-report`` and ``weather-calibrate --source log`` use to judge
whether the model actually beats the market.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class DataLog:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "log"
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, row_type: str, **fields: Any) -> None:
        now = datetime.now(UTC)
        row = {"t": now.isoformat(timespec="seconds"), "type": row_type, **fields}
        path = self.dir / f"{now:%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def rows(self, row_type: str | None = None) -> Iterator[dict[str, Any]]:
        for path in sorted(self.dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row_type is None or row.get("type") == row_type:
                        yield row
