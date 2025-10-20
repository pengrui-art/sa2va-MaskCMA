#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p: str | None):
    if not p:
        return None
    path = Path(p)
    if not path.exists():
        print(f"[WARN] file not found: {p}")
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"[WARN] failed to load {p}: {e}")
        return None


def pick_score(obj):
    """Return a single overall score for table.

    Priority:
    - if obj has key 'J&F' -> return it
    - elif obj has nested {'overall': {'JF': ...}} -> return it
    - elif obj has 'JF' -> return it
    - else None
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "J&F" in obj:
            return obj["J&F"]
        if (
            "overall" in obj
            and isinstance(obj["overall"], dict)
            and "JF" in obj["overall"]
        ):
            return obj["overall"]["JF"]
        if "JF" in obj:
            return obj["JF"]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate Ref-VOS style scores into a compact table"
    )
    ap.add_argument("--mevis_json", type=str, help="Output json from eval_mevis.py")
    ap.add_argument("--davis_json", type=str, help="Output json from eval_davis.py")
    ap.add_argument(
        "--refytvos_json", type=str, help="Output json for Ref-YTVOS (if available)"
    )
    ap.add_argument("--revos_json", type=str, help="Output json from eval_revos.py")
    args = ap.parse_args()

    rows = []
    entries = [
        ("MeViS", load_json(args.mevis_json)),
        ("Ref-DAVIS17", load_json(args.davis_json)),
        ("Ref-YTVOS", load_json(args.refytvos_json)),
        ("ReVOS", load_json(args.revos_json)),
    ]
    for name, obj in entries:
        score = pick_score(obj)
        rows.append((name, score))

    # Print a simple table
    header = ["Dataset", "Score (J&F or overall JF)"]
    w0 = max(len(header[0]), *(len(n) for n, _ in rows))
    w1 = max(
        len(header[1]),
        *(len(f"{s:.1f}") if isinstance(s, (int, float)) else 2 for _, s in rows),
    )

    def fmt(name, s):
        sval = f"{s:.1f}" if isinstance(s, (int, float)) else "-"
        return f"{name.ljust(w0)}  {sval.rjust(w1)}"

    print(f"{header[0].ljust(w0)}  {header[1].rjust(w1)}")
    print(f"{'-'*w0}  {'-'*w1}")
    for name, s in rows:
        print(fmt(name, s))


if __name__ == "__main__":
    main()
