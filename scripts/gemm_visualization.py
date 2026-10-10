"""Minimal, CPU-only GEMM diagrams shared by the four GIF scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 700
BG, PANEL, CELL, LINE = "#0c1220", "#141e30", "#202c42", "#34425a"
TEXT, MUTED = "#edf3ff", "#9cabc3"
A_COLOR, B_COLOR, C_COLOR = "#50d9cf", "#a99aff", "#8ce5a4"
GLOBAL_A = (131, 160, 16, 8, 14)
GLOBAL_B = (75, 460, 8, 16, 14)
GLOBAL_C = (955, 243, 16, 16, 17)


@dataclass(frozen=True)
class Demo:
    kind: str
    title: str
    kernel: str
    block: int
    depth: int
    micro: int

    @property
    def shared(self) -> bool:
        return self.kind != "naive"

    @property
    def vectorized(self) -> bool:
        return self.kind == "vectorized"


@dataclass(frozen=True)
class Step:
    phase: str
    tile: int = 0
    seconds: float = 1.0


def steps_for(demo: Demo) -> list[Step]:
    steps = [Step("intro", seconds=0.8)]
    for tile in range(8 // demo.depth if demo.shared else 1):
        steps.extend((Step("load", tile, 1.8), Step("compute", tile, 1.4)))
    last_tile = steps[-1].tile
    steps.extend((Step("store", last_tile, 1.4), Step("done", last_tile, 1.4)))
    return steps


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        str(Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf")),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def mix(color: str, amount: float, base: str = CELL) -> tuple[int, int, int]:
    return tuple(
        round(low + (high - low) * amount) for low, high in zip(rgb(base), rgb(color))
    )


def ease(progress: float) -> float:
    t = min(1.0, max(0.0, progress))
    return t * t * (3 - 2 * t)


def cells(rows: int, cols: int, row: int = 0, col: int = 0) -> set[tuple[int, int]]:
    return {(row + r, col + c) for r in range(rows) for c in range(cols)}


def bounds(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


class Canvas:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        self.draw = ImageDraw.Draw(self.image)

    def text(self, xy, label, size=20, color=TEXT, bold=False, anchor="mm"):
        self.draw.text(xy, label, font=font(size, bold), fill=color, anchor=anchor)

    def cell(self, x, y, size, color, amount=0.5):
        self.draw.rectangle(
            (x + 1, y + 1, x + size - 1, y + size - 1), fill=mix(color, amount)
        )

    def grid(self, layout, color, active=None, selected=None, amount=0.3):
        x, y, rows, cols, size = layout
        active, selected = active or set(), selected or set()
        for row in range(rows):
            for col in range(cols):
                strength = (
                    0.65
                    if (row, col) in selected
                    else amount
                    if (row, col) in active
                    else 0
                )
                self.cell(x + col * size, y + row * size, size, color, strength)

    def outline(self, layout, region, color, width=2):
        x, y, _, _, size = layout
        row, col, rows, cols = region
        self.draw.rectangle(
            (
                x + col * size,
                y + row * size,
                x + (col + cols) * size,
                y + (row + rows) * size,
            ),
            outline=color,
            width=width,
        )

    def arrow(self, start, end, color=LINE):
        self.draw.line((start, end), fill=color, width=2)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for offset in (-0.5, 0.5):
            self.draw.line(
                (
                    end,
                    (
                        end[0] - 10 * math.cos(angle + offset),
                        end[1] - 10 * math.sin(angle + offset),
                    ),
                ),
                fill=color,
                width=2,
            )


def operand_layout(demo: Demo, operand: str):
    if not demo.shared:
        rows, cols = (1, 8) if operand == "A" else (8, 1)
    elif operand == "A":
        rows, cols = (
            (demo.depth, demo.block) if demo.vectorized else (demo.block, demo.depth)
        )
    else:
        rows, cols = demo.depth, demo.block
    size = min(24, 270 // cols, 194 // rows)
    center_y = 272 if operand == "A" else 516
    return 600 - cols * size / 2, center_y - rows * size / 2, rows, cols, size


def source_region(demo: Demo, step: Step, operand: str):
    if not demo.shared:
        return cells(1, 8, 1, 0) if operand == "A" else cells(8, 1, 0, 1)
    offset = step.tile * demo.depth
    return (
        cells(demo.block, demo.depth, 0, offset)
        if operand == "A"
        else cells(demo.depth, demo.block, offset, 0)
    )


def used_region(demo: Demo, operand: str):
    if not demo.shared:
        return cells(1, 8) if operand == "A" else cells(8, 1)
    if operand == "A" and not demo.vectorized:
        return cells(demo.micro, demo.depth, demo.micro, 0)
    return cells(demo.depth, demo.micro, 0, demo.micro)


def load_tile(canvas: Canvas, demo: Demo, step: Step, operand: str, progress: float):
    """Move the whole region once; float4 lanes stay visibly grouped."""
    source = GLOBAL_A if operand == "A" else GLOBAL_B
    target = operand_layout(demo, operand)
    color = A_COLOR if operand == "A" else B_COLOR
    sx, sy, _, _, source_size = source
    tx, ty, _, _, target_size = target
    source_row, source_col, rows, cols = bounds(source_region(demo, step, operand))
    t = ease(min(1.0, progress / 0.85))
    size = source_size + (target_size - source_size) * t
    if operand == "A" and demo.vectorized:
        # Keep the four-lane groups together as the tile changes orientation.
        packet_size = round(size)
        packet = Image.new("RGBA", (cols * packet_size, rows * packet_size))
        draw = ImageDraw.Draw(packet)
        for row in range(rows):
            for col in range(cols):
                x, y = col * packet_size, row * packet_size
                draw.rectangle(
                    (x + 1, y + 1, x + packet_size - 1, y + packet_size - 1),
                    fill=mix(color, 0.65),
                )
            draw.rectangle(
                (
                    0,
                    row * packet_size,
                    cols * packet_size - 1,
                    (row + 1) * packet_size - 1,
                ),
                outline=color,
                width=2,
            )
        packet = packet.rotate(90 * t, resample=Image.Resampling.BICUBIC, expand=True)
        center_x = sx + (source_col + cols / 2) * source_size
        center_y = sy + (source_row + rows / 2) * source_size
        center_x += (tx + rows * target_size / 2 - center_x) * t
        center_y += (ty + cols * target_size / 2 - center_y) * t
        canvas.image.paste(
            packet,
            (round(center_x - packet.width / 2), round(center_y - packet.height / 2)),
            packet,
        )
        return
    positions = {}
    for row in range(rows):
        for col in range(cols):
            dest_row, dest_col = row, col
            x = sx + (source_col + col) * source_size
            y = sy + (source_row + row) * source_size
            x += (tx + dest_col * target_size - x) * t
            y += (ty + dest_row * target_size - y) * t
            canvas.cell(x, y, size, color, 0.65)
            positions[row, col] = (x, y)
    if demo.vectorized:
        for row in range(rows):
            for col in range(0, cols, 4):
                group = [positions[row, lane] for lane in range(col, col + 4)]
                left = min(x for x, _ in group)
                top = min(y for _, y in group)
                right = max(x for x, _ in group) + size
                bottom = max(y for _, y in group) + size
                canvas.draw.rectangle(
                    (left, top, right, bottom), outline=color, width=2
                )


def register_layout(demo: Demo):
    size = 44 if demo.micro == 1 else 26 if demo.micro == 2 else 19
    return (
        846 - demo.micro * size / 2,
        383 - demo.micro * size / 2,
        demo.micro,
        demo.micro,
        size,
    )


def store_tile(canvas: Canvas, demo: Demo, progress: float):
    source = register_layout(demo)
    sx, sy, rows, cols, source_size = source
    tx, ty, _, _, target_size = GLOBAL_C
    t = ease(min(1.0, progress / 0.85))
    size = source_size + (target_size - source_size) * t
    positions = {}
    for row in range(rows):
        for col in range(cols):
            end_x = tx + (demo.micro + col) * target_size
            end_y = ty + (demo.micro + row) * target_size
            x, y = sx + col * source_size, sy + row * source_size
            x += (end_x - x) * t
            y += (end_y - y) * t
            canvas.cell(x, y, size, C_COLOR, 0.65)
            positions[row, col] = x, y
    if demo.vectorized:
        for row in range(rows):
            x, y = positions[row, 0]
            canvas.draw.rectangle(
                (x, y, x + 4 * size, y + size), outline=C_COLOR, width=2
            )


def render(demo: Demo, step: Step, progress: float) -> Image.Image:
    canvas = Canvas()
    canvas.text((40, 36), demo.title, 34, TEXT, True, "lt")
    phase = {
        "intro": "",
        "load": "LOAD",
        "compute": "COMPUTE",
        "store": "STORE",
        "done": "",
    }
    canvas.text((1230, 50), phase[step.phase], 19, MUTED, anchor="rm")
    canvas.text((187, 118), "Global memory", 20, MUTED)
    canvas.text(
        (600, 118), "Shared memory" if demo.shared else "Thread operands", 20, MUTED
    )
    canvas.draw.rounded_rectangle((422, 145, 778, 636), radius=18, fill=PANEL)
    canvas.text((846, 320), "Registers", 18, MUTED)
    canvas.text((1091, 215), "C", 24, C_COLOR, True)

    for operand, source, color in (("A", GLOBAL_A, A_COLOR), ("B", GLOBAL_B, B_COLOR)):
        selected = source_region(demo, step, operand)
        canvas.grid(source, color, selected)
        canvas.outline(source, bounds(selected), color)
        x, y, rows, cols, size = source
        canvas.text((x + cols * size / 2, y - 20), operand, 24, color, True)
        target = operand_layout(demo, operand)
        tx, ty, tr, tc, ts = target
        loaded = step.phase in ("compute", "store", "done") or (
            step.phase == "load" and progress >= 0.85
        )
        focus = used_region(demo, operand) if step.phase == "compute" else set()
        canvas.grid(target, color, cells(tr, tc) if loaded else set(), focus)
        canvas.text(
            (600, ty - 20),
            f"{operand} tile"
            if demo.shared
            else f"{operand} {'row' if operand == 'A' else 'column'}",
            18,
            color,
        )
        canvas.arrow(
            (310, 272 if operand == "A" else 516), (410, 272 if operand == "A" else 516)
        )
        canvas.arrow((tx + tc * ts + 10, ty + tr * ts / 2), (803, 383))
        if demo.vectorized and loaded:
            # These marks survive after arrival, making the load width clear.
            if operand == "A":
                for col in range(tc):
                    canvas.outline(target, (0, col, 4, 1), color, 1)
            else:
                for row in range(tr):
                    for col in range(0, tc, 4):
                        canvas.outline(target, (row, col, 1, 4), color, 1)
        if step.phase == "load":
            load_tile(canvas, demo, step, operand, progress)

    registers = register_layout(demo)
    accumulated = step.phase in ("store", "done") or step.tile > 0
    active = (
        cells(demo.micro, demo.micro)
        if accumulated or step.phase == "compute"
        else set()
    )
    strength = 0.65 if accumulated else 0.25
    if step.phase == "compute":
        strength = (0.4 if step.tile else 0.2) + (0.25 if step.tile else 0.45) * ease(
            progress
        )
    canvas.grid(registers, C_COLOR, active, amount=strength)
    canvas.outline(registers, (0, 0, demo.micro, demo.micro), C_COLOR)
    canvas.arrow((891, 383), (940, 383))
    written = step.phase == "done" or (step.phase == "store" and progress >= 0.85)
    canvas.grid(
        GLOBAL_C,
        C_COLOR,
        cells(demo.micro, demo.micro, demo.micro, demo.micro) if written else set(),
        cells(demo.micro, demo.micro, demo.micro, demo.micro) if written else set(),
    )
    canvas.outline(GLOBAL_C, (demo.micro, demo.micro, demo.micro, demo.micro), C_COLOR)
    if step.phase == "store":
        store_tile(canvas, demo, progress)
    if demo.vectorized:
        canvas.text((600, 661), "float4", 18, MUTED)
    return canvas.image


def fixed_palette() -> Image.Image:
    accents = (TEXT, MUTED, A_COLOR, B_COLOR, C_COLOR)
    colors = [rgb(color) for color in (BG, PANEL, CELL, LINE, *accents)]
    for base in (BG, PANEL, CELL):
        for color in accents:
            colors.extend(mix(color, level / 12, base) for level in range(1, 13))
    colors = list(dict.fromkeys(colors))
    colors.extend([rgb(BG)] * (256 - len(colors)))
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in colors for channel in color])
    return palette


def run(demo: Demo) -> None:
    parser = argparse.ArgumentParser(
        description=f"Animate {demo.title.lower()} using Pillow; no CUDA required."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "gifs" / f"{demo.kernel}.gif"
    )
    parser.add_argument(
        "--width", type=int, default=WIDTH, help="output width (default: 1280)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="animation sampling rate, 1–50 (default: 12)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="also save a PNG contact sheet"
    )
    args = parser.parse_args()
    if not 640 <= args.width <= 2560:
        parser.error("--width must be between 640 and 2560")
    if not 1 <= args.fps <= 50:
        parser.error("--fps must be between 1 and 50")
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must have a .gif extension")
    steps = steps_for(demo)
    frames, durations = [], []
    palette = fixed_palette()
    for step in steps:
        count = max(1, round(step.seconds * args.fps))
        duration = max(20, round(step.seconds * 1000 / count / 10) * 10)
        for index in range(count):
            progress = index / (count - 1) if count > 1 else 1.0
            frame = render(demo, step, progress)
            if args.width != WIDTH:
                frame = frame.resize(
                    (args.width, round(HEIGHT * args.width / WIDTH)),
                    Image.Resampling.LANCZOS,
                )
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=1,
        optimize=False,
    )
    print(f"GIF: {args.output.resolve()} ({sum(durations) / 1000:.1f}s)")
    if args.preview:
        sheet = Image.new("RGB", (1280, 700), BG)
        for index, phase in enumerate(("intro", "load", "compute", "done")):
            step = next(step for step in steps if step.phase == phase)
            thumbnail = render(demo, step, 0.55).resize(
                (640, 350), Image.Resampling.LANCZOS
            )
            sheet.paste(thumbnail, ((index % 2) * 640, (index // 2) * 350))
        preview = args.output.with_suffix(".png")
        sheet.save(preview)
        print(f"Preview: {preview.resolve()}")
