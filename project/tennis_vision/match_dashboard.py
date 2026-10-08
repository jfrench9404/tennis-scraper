"""Offline match dashboard (one self-contained HTML file) built from a shot table.

Reads the folder written by ``tennis_vision.shot_table`` (``shot-table.json``,
cross-checked against ``shots.csv`` and ``points.csv``) and writes
``dashboard.html`` into a NEW folder. Nothing is fetched from the network: the
page has no CDN scripts or web fonts, and the table is inlined, so it opens from
file://.

The builder adds no tennis facts. It copies the table's rows (every value with
its ``_source`` and ``_basis``) and the table's provenance; the page only
filters, counts and draws them. In particular:

* Points, servers, winners and score text exist only if the table has them, and
  the table takes them only from John's labels.
* No spin or RPM. Ball speed is shown only where the table has it, as an estimate.
* Counts are coverage, never accuracy.

A shot links to ``replay.html`` only when the replay folder can be found, its
``replay-data.json`` is byte-identical to the one the table was built from
(sha256 in ``shot-table.json``), and ``replay.html`` exists. The link is stored
relative to the output folder.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

SCHEMA_VERSION = 1
TABLE_KIND = "shot_point_table"
SUPPORTED_TABLE_SCHEMAS = (1,)
TEMPLATE = Path(__file__).with_name("match_dashboard.html")
PLACEHOLDER = "__DASHBOARD_DATA__"
REPLAY_SEEK = ("replay.html does not seek from a link yet: the link opens the replay and the frame "
               "number is shown next to it. The '#frame=N' suffix is ignored by the current replay page.")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(folder):
    """Read and check a shot-table folder. Returns the parsed ``shot-table.json``."""
    folder = Path(folder)
    path = folder / "shot-table.json"
    if not path.is_file():
        raise ValueError(f"{folder} has no shot-table.json; pass a folder written by tennis_vision.shot_table")
    table = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or table.get("kind") != TABLE_KIND:
        raise ValueError(f"shot-table.json is not a {TABLE_KIND}")
    if table.get("schema_version") not in SUPPORTED_TABLE_SCHEMAS:
        raise ValueError(f"Unsupported shot table schema_version {table.get('schema_version')!r}")
    for key in ("shots", "points", "columns", "replay"):
        if key not in table:
            raise ValueError(f"shot-table.json is missing {key!r}")
    for kind in ("shots", "points"):
        columns = table["columns"][kind]
        for row in table[kind]:
            if set(row) != set(columns):
                raise ValueError(f"A {kind} row does not match the table's {kind} columns")
            for column in columns:
                if column.endswith("_source") and row[column] not in ("observed", "estimated", "human_confirmed",
                                                                         "unknown"):
                    raise ValueError(f"Invalid source {row[column]!r} in {kind}.{column}")
        csv_path = folder / f"{kind}.csv"
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                ids = [r[columns[0]] for r in reader]
                if reader.fieldnames != columns:
                    raise ValueError(f"{kind}.csv columns differ from shot-table.json")
            if ids != [str(r[columns[0]]) for r in table[kind]]:
                raise ValueError(f"{kind}.csv rows differ from shot-table.json")
    return table


def find_replay_link(table, output, replay=None, table_folder=None):
    """Relative link from ``output`` to the replay page, or None with a reason."""
    candidates = [Path(replay)] if replay else []
    folder_name = (table.get("replay") or {}).get("folder")
    if not replay and table_folder is not None and folder_name:
        candidates.append(Path(table_folder).parent / folder_name)
    expected = ((table.get("replay") or {}).get("input_sha256") or {}).get("replay-data.json")
    for folder in candidates:
        page, data = folder / "replay.html", folder / "replay-data.json"
        if not page.is_file():
            continue
        if not data.is_file() or not expected or _sha256(data) != expected:
            return None, f"{folder.name}/replay-data.json is not the one this table was built from; no replay links"
        try:
            relative = os.path.relpath(page.resolve(), Path(output).resolve())
        except ValueError:  # different drive on Windows
            return None, "replay.html is on a different drive from the dashboard; no replay links"
        return relative.replace(os.sep, "/"), "replay-data.json matches the table's input sha256"
    if replay:
        return None, f"{Path(replay)} has no replay.html; no replay links"
    return None, "replay folder not found next to the table (pass --replay); no replay links"


def build_payload(table, table_folder, replay_link, replay_reason):
    return {
        "schema_version": SCHEMA_VERSION, "kind": "match_dashboard",
        "table": {"folder": Path(table_folder).name, "sha256": _sha256(Path(table_folder) / "shot-table.json"),
                  "schema_version": table["schema_version"]},
        "replay": table["replay"], "labels": table.get("labels"), "calibration": table.get("calibration"),
        "conventions": table.get("conventions"), "counts": table.get("counts"),
        "limitations": table.get("limitations", []),
        "replay_link": {"href": replay_link, "status": replay_reason, "seek": REPLAY_SEEK if replay_link else None},
        "shots": table["shots"], "points": table["points"],
    }


def render(payload, template=TEMPLATE):
    text = Path(template).read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise ValueError("Dashboard template must contain the data placeholder exactly once")
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return text.replace(PLACEHOLDER, encoded)


def run(table_folder, output, replay=None):
    table_folder, output = Path(table_folder), Path(output)
    table = load_table(table_folder)
    if output.exists():
        raise ValueError("Output already exists; choose a new folder to preserve previous results")
    if output.resolve() == table_folder.resolve() or table_folder.resolve() in output.resolve().parents:
        raise ValueError("Write the dashboard to a new folder outside the table folder")
    if replay and (output.resolve() == Path(replay).resolve() or Path(replay).resolve() in output.resolve().parents):
        raise ValueError("Write the dashboard outside the replay folder; replay folders are not modified")
    output.mkdir(parents=True)
    link, reason = find_replay_link(table, output, replay, table_folder)
    payload = build_payload(table, table_folder, link, reason)
    (output / "dashboard.html").write_text(render(payload), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, required=True,
                        help="Folder written by tennis_vision.shot_table (shot-table.json, shots.csv, points.csv)")
    parser.add_argument("--output", type=Path, required=True, help="NEW folder for dashboard.html")
    parser.add_argument("--replay", type=Path,
                        help="Replay folder the table was built from (default: the table's replay folder name, "
                             "next to the table folder). Used only for shot links.")
    args = parser.parse_args(argv)
    payload = run(args.table, args.output, args.replay)
    print(json.dumps({"output": str(args.output / "dashboard.html"), "shots": len(payload["shots"]),
                      "points": len(payload["points"]), "replay_link": payload["replay_link"]}, indent=2))


if __name__ == "__main__":
    main()
