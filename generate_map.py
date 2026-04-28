"""Warehouse map generator for multi-agent pathfinding simulations.

Cell encoding:
- 0: empty floor / aisle
- 1: rack (obstacle)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

try:
	import numpy as np
except ImportError as exc:
	raise ImportError(
		"Numpy is required. Install dependencies with: pip install -r requirements.txt"
	) from exc


@dataclass(frozen=True)
class MapConfig:
	"""Configuration for generating a warehouse map."""

	width: int = 30
	height: int = 20
	rack_width: int = 3
	rack_height: int = 2
	random_layout: bool = False
	rack_count: int = 24
	aisle: int = 1
	seed: Optional[int] = None


def _validate_config(config: MapConfig) -> None:
	if config.width <= 0 or config.height <= 0:
		raise ValueError("Map width and height must be positive.")
	if config.rack_width <= 0 or config.rack_height <= 0:
		raise ValueError("Rack width and rack height must be positive.")
	if config.rack_width > config.width or config.rack_height > config.height:
		raise ValueError("Rack size cannot be larger than map size.")
	if config.aisle < 0:
		raise ValueError("Aisle must be >= 0.")
	if config.rack_count < 0:
		raise ValueError("Rack count must be >= 0.")


def _can_place_rack(
	grid: np.ndarray,
	top: int,
	left: int,
	rack_h: int,
	rack_w: int,
	aisle: int,
) -> bool:
	h, w = grid.shape
	end_row = top + rack_h
	end_col = left + rack_w
	if end_row > h or end_col > w:
		return False

	r0 = max(0, top - aisle)
	c0 = max(0, left - aisle)
	r1 = min(h, end_row + aisle)
	c1 = min(w, end_col + aisle)
	return bool(np.all(grid[r0:r1, c0:c1] == 0))


def _place_rack(grid: np.ndarray, top: int, left: int, rack_h: int, rack_w: int) -> None:
	grid[top : top + rack_h, left : left + rack_w] = 1


def _generate_fixed_layout(grid: np.ndarray, config: MapConfig) -> int:
	"""Generate structured shelves with aisles in a regular pattern."""
	h, w = grid.shape
	step_y = config.rack_height + config.aisle
	step_x = config.rack_width + config.aisle
	placed = 0

	for top in range(1, h - config.rack_height + 1, step_y):
		for left in range(1, w - config.rack_width + 1, step_x):
			_place_rack(grid, top, left, config.rack_height, config.rack_width)
			placed += 1

	return placed


def _generate_random_layout(grid: np.ndarray, config: MapConfig) -> int:
	"""Generate random non-overlapping rack positions with aisle spacing."""
	rng = np.random.default_rng(config.seed)
	h, w = grid.shape
	candidates = [
		(top, left)
		for top in range(0, h - config.rack_height + 1)
		for left in range(0, w - config.rack_width + 1)
	]
	rng.shuffle(candidates)

	placed = 0
	for top, left in candidates:
		if placed >= config.rack_count:
			break
		if _can_place_rack(
			grid,
			top,
			left,
			config.rack_height,
			config.rack_width,
			config.aisle,
		):
			_place_rack(grid, top, left, config.rack_height, config.rack_width)
			placed += 1

	return placed


def generate_warehouse_map(config: MapConfig) -> tuple[np.ndarray, int]:
	"""Create a warehouse map and return (grid, number_of_racks_placed)."""
	_validate_config(config)
	grid = np.zeros((config.height, config.width), dtype=np.uint8)

	if config.random_layout:
		placed = _generate_random_layout(grid, config)
	else:
		placed = _generate_fixed_layout(grid, config)

	return grid, placed


def plot_map(grid: np.ndarray, title: str = "Warehouse Map", save_path: Optional[str] = None) -> None:
	"""Display and optionally save a generated map."""
	try:
		import matplotlib.pyplot as plt
		from matplotlib.colors import ListedColormap
	except ImportError as exc:
		raise ImportError(
			"Matplotlib is required for plotting. Install dependencies with: pip install -r requirements.txt"
		) from exc

	cmap = ListedColormap(["#f5f5f5", "#2f4f4f"])
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.imshow(grid, cmap=cmap, interpolation="none", origin="upper")
	ax.set_title(title)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
	ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
	ax.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
	ax.tick_params(which="both", bottom=False, left=False, labelbottom=True, labelleft=True)
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=180)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate a warehouse map for pathfinding simulation.")
	parser.add_argument("--width", type=int, default=30, help="Map width (columns).")
	parser.add_argument("--height", type=int, default=20, help="Map height (rows).")
	parser.add_argument("--rack-width", type=int, default=3, help="Rack width.")
	parser.add_argument("--rack-height", type=int, default=2, help="Rack height.")
	parser.add_argument("--random", action="store_true", help="Use random rack layout.")
	parser.add_argument("--rack-count", type=int, default=24, help="Rack count for random layout.")
	parser.add_argument("--aisle", type=int, default=1, help="Minimum empty-cell spacing between racks.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible random layout.")
	parser.add_argument("--save", type=str, default=None, help="Optional path to save map image.")
	parser.add_argument("--show", dest="show", action="store_true", help="Show map in a new matplotlib window (default: enabled).")
	parser.add_argument("--no-show", dest="show", action="store_false", help="Disable map window display.")
	parser.set_defaults(show=True)
	parser.add_argument(
		"--print-grid",
		action="store_true",
		help="Print map array to terminal (0 = empty, 1 = rack).",
	)
	return parser


def main() -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		plt = None

	parser = _build_parser()
	args = parser.parse_args()

	config = MapConfig(
		width=args.width,
		height=args.height,
		rack_width=args.rack_width,
		rack_height=args.rack_height,
		random_layout=args.random,
		rack_count=args.rack_count,
		aisle=args.aisle,
		seed=args.seed,
	)

	grid, placed = generate_warehouse_map(config)
	print(f"Generated map: {config.width}x{config.height}")
	print(f"Rack size: {config.rack_width}x{config.rack_height}")
	print(f"Layout mode: {'random' if config.random_layout else 'fixed'}")
	if config.random_layout and placed < config.rack_count:
		print(
			f"Requested {config.rack_count} racks, but only {placed} can fit with aisle={config.aisle}."
		)
	else:
		print(f"Racks placed: {placed}")

	# if args.print_grid:
	# print(grid)
	print(placed)

	if args.show or args.save:
		title = f"Warehouse Map ({'Random' if config.random_layout else 'Fixed'})"
		plot_map(grid, title=title, save_path=args.save)
		if args.show and plt is not None:
			plt.show()


if __name__ == "__main__":
	main()
