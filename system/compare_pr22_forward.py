#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "pr22_compare"
BASE = ROOT / "base"
HEAD = ROOT / "head"
HEAD_LEAN = ROOT / "head_lean"
OUT = ROOT / "compare.md"


def load_json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def by_name(rows, key):
    return {row[key]: row for row in rows}


def pct_delta(head, base):
    if base == 0:
        return None
    return (head - base) / base * 100.0


def fmt_pct(v):
    if v is None:
        return "n/a"
    return f"{v:+.1f}%"


def fmt_delta(head, base, digits=1):
    return f"{head - base:+.{digits}f}"


def main():
    base_scale = by_name(load_json(BASE / "scale_ladder.json"), "name")
    head_scale = by_name(load_json(HEAD / "scale_ladder.json"), "name")
    base_layer = by_name(load_json(BASE / "layer_breakdown.json"), "name")
    head_layer = by_name(load_json(HEAD / "layer_breakdown.json"), "name")
    head_lean = by_name(load_json(HEAD_LEAN / "scale_ladder.json"), "name")

    scales = ["5B", "13B", "20B"]

    lines = []
    lines.append("# PR #22 Forward Comparison")
    lines.append("")

    lines.append("## Single-stream common-path comparison")
    lines.append("")
    lines.append("| scale | base tok/s | head tok/s | tok/s delta | base fwd(ms) | head fwd(ms) | fwd delta | base peak RSS(MB) | head peak RSS(MB) | RSS delta | head FFN mode |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for scale in scales:
        b = base_scale.get(scale)
        h = head_scale.get(scale)
        if not b or not h:
            lines.append(f"| {scale} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
            continue
        lines.append(
            "| {} | {:.1f} | {:.1f} | {} | {:.1f} | {:.1f} | {} ms | {:.0f} | {:.0f} | {} MB | {} |".format(
                scale,
                b["tok_per_s"],
                h["tok_per_s"],
                fmt_pct(pct_delta(h["tok_per_s"], b["tok_per_s"])),
                b["median_fwd_ms"],
                h["median_fwd_ms"],
                fmt_delta(h["median_fwd_ms"], b["median_fwd_ms"]),
                b["rss_mb_peak_timed"],
                h["rss_mb_peak_timed"],
                fmt_delta(h["rss_mb_peak_timed"], b["rss_mb_peak_timed"], 0),
                h.get("ffn_mode", "unknown"),
            )
        )
    lines.append("")

    lines.append("## Layer bucket deltas")
    lines.append("")
    lines.append("| scale | base ane% | head ane% | ane delta | base stage% | head stage% | stage delta | base read% | head read% | read delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scale in ["5B", "13B"]:
        b = base_layer.get(scale)
        h = head_layer.get(scale)
        if not b or not h or not b.get("success", True) or not h.get("success", True):
            lines.append(f"| {scale} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
            continue
        lines.append(
            "| {} | {:.1f}% | {:.1f}% | {} pp | {:.1f}% | {:.1f}% | {} pp | {:.1f}% | {:.1f}% | {} pp |".format(
                scale,
                b["ane_pct"],
                h["ane_pct"],
                fmt_delta(h["ane_pct"], b["ane_pct"]),
                b["stage_pct"],
                h["stage_pct"],
                fmt_delta(h["stage_pct"], b["stage_pct"]),
                b["read_pct"],
                h["read_pct"],
                fmt_delta(h["read_pct"], b["read_pct"]),
            )
        )
    lines.append("")

    if head_lean:
        lines.append("## Head-only lean forward addendum")
        lines.append("")
        lines.append("| scale | common tok/s | lean tok/s | lean delta | common peak RSS(MB) | lean peak RSS(MB) | lean RSS delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for scale in scales:
            h = head_scale.get(scale)
            l = head_lean.get(scale)
            if not h or not l:
                lines.append(f"| {scale} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
                continue
            lines.append(
                "| {} | {:.1f} | {:.1f} | {} | {:.0f} | {:.0f} | {} MB |".format(
                    scale,
                    h["tok_per_s"],
                    l["tok_per_s"],
                    fmt_pct(pct_delta(l["tok_per_s"], h["tok_per_s"])),
                    h["rss_mb_peak_timed"],
                    l["rss_mb_peak_timed"],
                    fmt_delta(l["rss_mb_peak_timed"], h["rss_mb_peak_timed"], 0),
                )
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `head FFN mode` is reported from the head-side scale results.")
    lines.append("- Kernel wall-vs-hw results are intentionally excluded from the delta table until non-zero hardware timing is available.")
    lines.append("")

    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
