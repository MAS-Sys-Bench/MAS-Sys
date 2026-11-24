"""Shared helpers extracted from the original workflow study (MIT)."""

from __future__ import annotations

import asyncio
from html import escape
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
DEFAULT_PYTHON = sys.executable


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def pctl(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def wait_ready(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{base_url}/models")
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
            await asyncio.sleep(2.0)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s: {last_error}")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    log_fh = getattr(process, "_motivation_log_fh", None)
    if log_fh is not None:
        log_fh.close()


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def build_context(target_tokens: int) -> str:
    # Approx 11-13 tokens per row for Qwen tokenizers.  The exact value is
    # recorded as a rough synthetic setting, not used as a correctness oracle.
    rows = []
    for i in range(max(1, target_tokens // 12)):
        rows.append(
            "Fact %04d: subsystem alpha=%d beta=%d queue=%d cache_marker=CTX%04d."
            % (i, i % 17, i % 31, i % 7, i)
        )
    return "\n".join(rows)


def topology_color(topology: str, index: int) -> str:
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#c2410c",
    ]
    return palette[index % len(palette)]


def write_rank_bump_svg(
    path: Path,
    rank_rows: list[dict[str, Any]],
    *,
    condition_order: list[str],
    topology_order: list[str],
) -> None:
    if not rank_rows:
        return
    max_rank = max(int(row["rank"]) for row in rank_rows)
    width = max(980, 150 * len(condition_order) + 260)
    height = max(520, 58 * max_rank + 160)
    left = 100
    right = 210
    top = 84
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_gap = plot_w / max(1, len(condition_order) - 1)
    y_gap = plot_h / max(1, max_rank - 1)
    lookup = {(row["condition"], row["topology"]): row for row in rank_rows}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Topology latency ranking across serving conditions</text>',
        '<text x="28" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Rank 1 is fastest by median end-to-end latency. Labels show topology variants.</text>',
    ]

    for rank in range(1, max_rank + 1):
        y = top + (rank - 1) * y_gap
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 18}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{rank}</text>'
        )

    for idx, condition in enumerate(condition_order):
        x = left + idx * x_gap
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - bottom + 8}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 58}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{escape(condition)}</text>'
        )

    for topo_idx, topology in enumerate(topology_order):
        points = []
        point_rows = []
        for cond_idx, condition in enumerate(condition_order):
            row = lookup.get((condition, topology))
            if not row:
                continue
            x = left + cond_idx * x_gap
            y = top + (int(row["rank"]) - 1) * y_gap
            points.append((x, y))
            point_rows.append(row)
        if not points:
            continue
        color = topology_color(topology, topo_idx)
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for (x, y), row in zip(points, point_rows):
            title = (
                f"{topology} @ {row['condition']}: "
                f"rank {row['rank']}, median {row['median_s']}s, p90 {row['p90_s']}s"
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"><title>{escape(title)}</title></circle>'
            )
        last_x, last_y = points[-1]
        parts.append(
            f'<text x="{last_x + 10:.1f}" y="{last_y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="{color}">{escape(topology)}</text>'
        )

    parts.append(
        f'<text x="{left - 62}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#374151">rank</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_rank_heatmap_svg(
    path: Path,
    rank_rows: list[dict[str, Any]],
    *,
    condition_order: list[str],
    topology_order: list[str],
) -> None:
    if not rank_rows:
        return
    cell_w = 132
    cell_h = 42
    left = 170
    top = 92
    width = left + cell_w * len(condition_order) + 36
    height = top + cell_h * len(topology_order) + 64
    lookup = {(row["condition"], row["topology"]): row for row in rank_rows}
    max_rank = max(int(row["rank"]) for row in rank_rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Ranking heatmap</text>',
        '<text x="28" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Each cell shows rank / median seconds. Darker green means faster rank.</text>',
    ]

    for col, condition in enumerate(condition_order):
        x = left + col * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#374151">{escape(condition)}</text>'
        )

    for row_idx, topology in enumerate(topology_order):
        y = top + row_idx * cell_h
        parts.append(
            f'<text x="{left - 12}" y="{y + 26}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#374151">{escape(topology)}</text>'
        )
        for col, condition in enumerate(condition_order):
            x = left + col * cell_w
            row = lookup.get((condition, topology))
            if row:
                rank = int(row["rank"])
                intensity = 1 - ((rank - 1) / max(1, max_rank - 1))
                green = int(245 - intensity * 130)
                red = int(245 - intensity * 90)
                blue = int(245 - intensity * 120)
                fill = f"rgb({red},{green},{blue})"
                label = f"#{rank} / {row['median_s']}s"
                title = (
                    f"{topology} @ {condition}: "
                    f"rank {rank}, median {row['median_s']}s, p90 {row['p90_s']}s"
                )
            else:
                fill = "#f9fafb"
                label = ""
                title = f"{topology} @ {condition}: missing"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h - 6}" rx="4" fill="{fill}" stroke="#ffffff"><title>{escape(title)}</title></rect>'
            )
            parts.append(
                f'<text x="{x + (cell_w - 6) / 2:.1f}" y="{y + 23}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{escape(label)}</text>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
