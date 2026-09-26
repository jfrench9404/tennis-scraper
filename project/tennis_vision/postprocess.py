"""Create a consolidated shot file from an existing events.jsonl export."""
import argparse
import json
from pathlib import Path

from .events import build_shots


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse raw tennis event candidates into shots.")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_events = []
    with args.events.open(encoding="utf-8") as stream:
        for line in stream:
            raw_events.extend(json.loads(line).get("events", []))
    result = build_shots(raw_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["consolidated_counts"]))


if __name__ == "__main__":
    main()
