"""Append-only query log (JSON Lines). Raw material for the golden eval set."""

import json
from datetime import UTC, datetime

from . import config  # module attribute read at call time, so tests can redirect LOG_FILE


def log_query(result: dict, store_name: str, doc_id: str) -> None:
    path = config.LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(UTC).isoformat(), "store": store_name, "doc_id": doc_id, **result}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
