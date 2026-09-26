#!/usr/bin/env python
"""Summarise overfull/underfull boxes in a LaTeX log.

Why this exists: an ad-hoc shell one-liner that stripped non-digits from
"Overfull \\hbox (4.27965pt too wide)" was reporting harmless 4pt boxes as
"severe (>50pt)", which made a clean build look broken. Measure properly.

    python submission/check_overfull.py [jobname ...]      # default: manuscript proof
"""

from __future__ import annotations

import pathlib
import re
import sys

OVERFULL = re.compile(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\) in (.*)")
# Springer Nature's text block is narrower than the article-class proof, so a box that is fine in
# proof.pdf can overflow in manuscript.pdf. Anything past ~10pt is visible in print.
THRESHOLDS = (50.0, 20.0, 10.0, 5.0)


def summarise(log_path: pathlib.Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hits = [(float(pt), where.strip()) for pt, where in OVERFULL.findall(text)]
    hits.sort(key=lambda h: -h[0])

    print(f"{log_path.name}: {len(hits)} overfull hbox(es)")
    if not hits:
        return 0
    for t in THRESHOLDS:
        print(f"   >{t:g}pt: {sum(1 for v, _ in hits if v > t)}")
    print("   worst:")
    for value, where in hits[:10]:
        print(f"     {value:8.1f}pt  {where[:90]}")
    return sum(1 for v, _ in hits if v > THRESHOLDS[0])


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    jobs = sys.argv[1:] or ["manuscript", "proof"]
    severe = 0
    for job in jobs:
        log = root / f"{job}.log"
        if not log.exists():
            print(f"{job}.log: not built")
            continue
        severe += summarise(log)
        print()
    return 1 if severe else 0


if __name__ == "__main__":
    sys.exit(main())
