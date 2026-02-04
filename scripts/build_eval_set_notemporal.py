from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.query_expansion import is_relative_time_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_set_curated_last12m.json")
    parser.add_argument("--output", default="data/eval_set_curated_last12m_notemporal.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = json.loads(input_path.read_text())
    filtered = []
    for entry in data:
        question = (entry or {}).get("question") if isinstance(entry, dict) else None
        if not question:
            continue
        if is_relative_time_query(question):
            continue
        filtered.append(dict(entry))

    for idx, entry in enumerate(filtered, start=1):
        entry["id"] = idx

    output_path = Path(args.output)
    output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2))
    print(f"Wrote {len(filtered)} entries to {output_path}")


if __name__ == "__main__":
    main()
