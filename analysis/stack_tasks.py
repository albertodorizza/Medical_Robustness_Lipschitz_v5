"""
analysis/stack_tasks.py
=======================
Stack the per-task figures of an insight into the ONE figure the paper prints:
classification on top, segmentation underneath, a single key below both.

    python -m analysis.stack_tasks \\
        paper_results/depth_leakage_classification_row.pdf \\
        paper_results/depth_leakage_segmentation_row.pdf \\
        -o paper_results/fig1_depth_leakage.pdf

    python -m analysis.stack_tasks --pairs paper_results        # all of them


Why a composer and not a two-row figure function
------------------------------------------------
The two rows come from two runs of the same script over two different results
trees -- different models, different datasets, different sweeps -- and making
one process draw both would mean loading both sweeps in one go and threading a
row index through seven ``figure`` functions whose panel layouts have nothing
in common (one is a grid over architectures, one is three dumbbell panels, one
is a pair of dose-response curves). None of that changes a number in the
figures; all of it is layout. Composing the finished PDFs keeps each script
responsible for one row and this file responsible for the page.

What the scripts have to do for it is therefore small, and it is two flags that
:mod:`analysis.common` gives all of them for free:

    --no_legend    on the TOP row, so the key is drawn once, under the bottom
                   row, where it reads as belonging to the whole figure;
    --row_label    written down the left of each row, so the reader is told
                   which task a row is without a title inside a panel.

The two rows must have the same number of panels in the same order, which is
what ``1_depth_leakage.py --layout row`` exists to guarantee for the one script
whose natural classification layout is a grid.


What this does to the page
--------------------------
Nothing but placement. The pages are copied unscaled and centred on a canvas as
wide as the widest of them; no resampling, no re-encoding of the vector
content, no font surgery, so the text is the same size in the composite as it
was in the inputs -- which is the point of drawing both rows at the same figure
width in the first place. If the two rows come out at different widths, that is
a real mismatch between the two runs and it shows up as a centred row rather
than being silently scaled away.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from pypdf import PageObject

# The order the rows are stacked in, when the pairs are discovered rather than
# named: classification first, because every insight is stated for it first.
TASK_ORDER = ["classification", "segmentation"]

# Gap between rows, in points.  Each input already carries its own tight
# bounding box, so this is the only whitespace the composite adds.
GAP = 6.0


def stack(paths: list[str], out: str, gap: float = GAP) -> str:
    """Place ``paths`` one above the other, in order, on a single page."""
    pages = []
    for p in paths:
        reader = PdfReader(p)
        if len(reader.pages) != 1:
            sys.exit(f"{p} has {len(reader.pages)} pages; expected one figure")
        pages.append(reader.pages[0])

    widths = [float(pg.mediabox.width) for pg in pages]
    heights = [float(pg.mediabox.height) for pg in pages]
    width = max(widths)
    height = sum(heights) + gap * (len(pages) - 1)

    canvas = PageObject.create_blank_page(width=width, height=height)
    y = height
    for pg, w, h in zip(pages, widths, heights):
        y -= h
        # Each source page has its own origin; shift it so its lower-left
        # corner lands where this row starts, and centre it if it is narrower.
        dx = (width - w) / 2.0 - float(pg.mediabox.left)
        dy = y - float(pg.mediabox.bottom)
        canvas.merge_transformed_page(pg, Transformation().translate(dx, dy))
        y -= gap
    canvas.mediabox = RectangleObject((0, 0, width, height))

    writer = PdfWriter()
    writer.add_page(canvas)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as fh:
        writer.write(fh)
    print(f"  [fig] {out}  ({' + '.join(os.path.basename(p) for p in paths)})")
    return out


def discover(directory: str) -> dict[str, dict[str, str]]:
    """Find ``<stem>_<task>[_<suffix>].pdf`` pairs in a results directory.

    The stem of every script already carries the task, so a pair is two files
    whose names differ only in that word.  Anything that does not pair up is
    left alone and reported, since a missing half is usually a sweep that has
    not been run rather than a naming accident.
    """
    found: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^(?P<stem>.+?)_(?P<task>classification|segmentation)"
                         r"(?P<rest>.*)\.pdf$")
    for name in sorted(os.listdir(directory)):
        m = pattern.match(name)
        if not m:
            continue
        key = m.group("stem") + m.group("rest")
        found.setdefault(key, {})[m.group("task")] = os.path.join(directory, name)
    return found


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stack per-task insight figures into one paper figure.")
    p.add_argument("inputs", nargs="*",
                   help="PDFs to stack, top first. Omit with --pairs.")
    p.add_argument("-o", "--out", default=None, help="output PDF.")
    p.add_argument("--pairs", default=None, metavar="DIR",
                   help="stack every classification/segmentation pair found in "
                        "DIR, writing <stem>_stacked.pdf next to them.")
    p.add_argument("--gap", type=float, default=GAP,
                   help=f"whitespace between rows, in points. Default {GAP:g}.")
    args = p.parse_args()

    if args.pairs:
        pairs = discover(args.pairs)
        if not pairs:
            sys.exit(f"no per-task figures found in {args.pairs}")
        for key, byt in pairs.items():
            have = [t for t in TASK_ORDER if t in byt]
            if len(have) < 2:
                print(f"  [skip] {key}: only {have[0]} present")
                continue
            stack([byt[t] for t in have],
                  os.path.join(args.pairs, f"{key}_stacked.pdf"), args.gap)
        return

    if len(args.inputs) < 2 or not args.out:
        sys.exit("give at least two input PDFs and -o OUT, or use --pairs DIR")
    for path in args.inputs:
        if not os.path.exists(path):
            sys.exit(f"missing input: {path}")
    stack(args.inputs, args.out, args.gap)


if __name__ == "__main__":
    main()
