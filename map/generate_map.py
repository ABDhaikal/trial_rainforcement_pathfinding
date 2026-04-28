"""
=============================================================
  Warehouse Multi-Forklift Environment — Interactive Setup
  Global Path Planning — Reinforcement Learning Research
=============================================================
  Fitur:
    1. Input ukuran peta (custom rows x cols)
    2. Input lokasi rak (random / default / manual)
    3. Tentukan ukuran rak (1x1, 1x2, 2x2, dst)
    4. Letakkan posisi forklift secara interaktif
    5. Animasi step pergerakan forklift

  Requirements:
    pip install numpy matplotlib gymnasium
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox, RadioButtons
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
import random
import sys

# ─────────────────────────────────────────────
#  KONSTANTA CELL TYPE
# ─────────────────────────────────────────────
FREE     = 0
WALL     = 1
RACK     = 2
PICKUP   = 3
DROPOFF  = 4
FORKLIFT = 5
CHARGING = 6

# Action codes
STAY  = 0; UP    = 1; DOWN  = 2; LEFT  = 3; RIGHT = 4
N_ACTIONS = 5
DELTA = { STAY:(0,0), UP:(-1,0), DOWN:(1,0), LEFT:(0,-1), RIGHT:(0,1) }

# Warna visualisasi
CELL_COLORS = {
    FREE    : "#1a2332",
    WALL    : "#374151",
    RACK    : "#1e3a5f",
    PICKUP  : "#14532d",
    DROPOFF : "#4c1d1d",
    FORKLIFT: "#312e81",
    CHARGING: "#3b2a00",
}
CELL_LABELS = {
    WALL    : ("■",  "#9ca3af"),
    RACK    : ("▣",  "#3b82f6"),
    PICKUP  : ("P",  "#4ade80"),
    DROPOFF : ("D",  "#f87171"),
    FORKLIFT: ("FK", "#c4b5fd"),
    CHARGING: ("⚡", "#fbbf24"),
}
FORKLIFT_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#06b6d4", "#ec4899"]


# ─────────────────────────────────────────────
#  1. KONFIGURASI SETUP — Wizard Input
# ─────────────────────────────────────────────

class WarehouseConfig:
    """
    Kelas untuk menyimpan semua konfigurasi warehouse
    yang diinput oleh user.
    """
    def __init__(self):
        self.rows         = 12
        self.cols         = 12
        self.n_forklifts  = 3
        self.rack_mode    = "default"   # "default" | "random" | "manual"
        self.rack_size    = (2, 2)      # (height, width) per blok rak
        self.n_rack_blocks = 6          # jumlah blok rak (untuk random)
        self.forklift_positions = []    # list of (row, col)
        self.pickup_positions   = []
        self.dropoff_positions  = []
        self.grid = None                # numpy array final

    def __repr__(self):
        return (f"WarehouseConfig(rows={self.rows}, cols={self.cols}, "
                f"forklifts={self.n_forklifts}, rack_mode={self.rack_mode}, "
                f"rack_size={self.rack_size})")


def input_wizard() -> WarehouseConfig:
    """Wizard CLI untuk input konfigurasi warehouse secara interaktif."""
    cfg = WarehouseConfig()
    print("\n" + "=" * 55)
    print("  WAREHOUSE SETUP WIZARD")
    print("=" * 55)

    # ── Step 1: Ukuran Peta ──
    print("\n[1/5] UKURAN PETA")
    while True:
        try:
            rows = int(input("  Masukkan jumlah baris  (min 8, default 12): ").strip() or "12")
            cols = int(input("  Masukkan jumlah kolom  (min 8, default 12): ").strip() or "12")
            if rows >= 8 and cols >= 8:
                cfg.rows = rows
                cfg.cols = cols
                print(f"  ✅ Ukuran peta: {rows} x {cols}")
                break
            print("  ⚠ Ukuran minimal 8x8, coba lagi.")
        except ValueError:
            print("  ⚠ Input harus angka.")

    # ── Step 2: Jumlah Forklift ──
    print("\n[2/5] JUMLAH FORKLIFT")
    while True:
        try:
            n = int(input("  Jumlah forklift (1-6, default 3): ").strip() or "3")
            if 1 <= n <= 6:
                cfg.n_forklifts = n
                print(f"  ✅ Jumlah forklift: {n}")
                break
            print("  ⚠ Jumlah forklift 1-6.")
        except ValueError:
            print("  ⚠ Input harus angka.")

    # ── Step 3: Mode Rak ──
    print("\n[3/5] MODE PENEMPATAN RAK")
    print("  1. Default  — rak berjajar rapi (recommended)")
    print("  2. Random   — rak ditempatkan acak")
    print("  3. Manual   — input koordinat rak sendiri")
    while True:
        mode = input("  Pilih mode (1/2/3, default 1): ").strip() or "1"
        if mode in ["1", "2", "3"]:
            cfg.rack_mode = {"1": "default", "2": "random", "3": "manual"}[mode]
            print(f"  ✅ Mode rak: {cfg.rack_mode}")
            break
        print("  ⚠ Pilih 1, 2, atau 3.")

    # ── Step 4: Ukuran Rak ──
    print("\n[4/5] UKURAN BLOK RAK")
    print("  Contoh: 1x1, 1x2, 2x2, 2x3")
    while True:
        try:
            rh = int(input("  Tinggi rak / rows (1-4, default 2): ").strip() or "2")
            rw = int(input("  Lebar  rak / cols (1-4, default 2): ").strip() or "2")
            if 1 <= rh <= 4 and 1 <= rw <= 4:
                cfg.rack_size = (rh, rw)
                print(f"  ✅ Ukuran rak: {rh} x {rw}")
                break
            print("  ⚠ Ukuran rak antara 1-4.")
        except ValueError:
            print("  ⚠ Input harus angka.")

    if cfg.rack_mode == "random":
        while True:
            try:
                nb = int(input("  Jumlah blok rak random (default 6): ").strip() or "6")
                if nb >= 1:
                    cfg.n_rack_blocks = nb
                    print(f"  ✅ Jumlah blok rak: {nb}")
                    break
            except ValueError:
                print("  ⚠ Input harus angka.")

    # ── Step 5: Posisi Forklift ──
    print("\n[5/5] POSISI FORKLIFT")
    print(f"  Peta berukuran {cfg.rows}x{cfg.cols}.")
    print(f"  Koordinat valid: row 1-{cfg.rows-2}, col 1-{cfg.cols-2}")
    print("  (baris & kolom 0 serta ujung adalah wall)")

    if input("  Gunakan posisi forklift otomatis? (y/n, default y): ").strip().lower() in ["", "y"]:
        cfg.forklift_positions = _auto_forklift_positions(cfg)
        print(f"  ✅ Posisi otomatis: {cfg.forklift_positions}")
    else:
        cfg.forklift_positions = []
        for i in range(cfg.n_forklifts):
            while True:
                try:
                    r = int(input(f"  Forklift {i+1} — baris (row): "))
                    c = int(input(f"  Forklift {i+1} — kolom (col): "))
                    if (1 <= r <= cfg.rows - 2 and 1 <= c <= cfg.cols - 2
                            and [r, c] not in cfg.forklift_positions):
                        cfg.forklift_positions.append((r, c))
                        print(f"  ✅ Forklift {i+1} → ({r}, {c})")
                        break
                    print("  ⚠ Posisi tidak valid atau sudah dipakai.")
                except ValueError:
                    print("  ⚠ Input harus angka.")

    print("\n  ✅ Konfigurasi selesai! Generating map...\n")
    return cfg


def _auto_forklift_positions(cfg: WarehouseConfig) -> list:
    """Generate posisi forklift otomatis tersebar merata di baris atas."""
    positions = []
    step = max(1, (cfg.cols - 2) // cfg.n_forklifts)
    for i in range(cfg.n_forklifts):
        c = 1 + i * step
        if c >= cfg.cols - 1:
            c = cfg.cols - 2
        positions.append((1, c))
    return positions


# ─────────────────────────────────────────────
#  2. MAP GENERATOR
# ─────────────────────────────────────────────

class MapGenerator:
    """Generate grid map berdasarkan konfigurasi."""

    def __init__(self, cfg: WarehouseConfig):
        self.cfg = cfg

    def generate(self) -> np.ndarray:
        """Main method: buat grid sesuai config."""
        grid = self._make_empty_grid()
        if self.cfg.rack_mode == "default":
            grid = self._place_default_racks(grid)
        elif self.cfg.rack_mode == "random":
            grid = self._place_random_racks(grid)
        elif self.cfg.rack_mode == "manual":
            grid = self._place_manual_racks(grid)
        grid = self._place_pickup_dropoff(grid)
        grid = self._place_forklifts(grid)
        return grid

    def _make_empty_grid(self) -> np.ndarray:
        """Buat grid kosong dengan border wall."""
        grid = np.full((self.cfg.rows, self.cfg.cols), FREE, dtype=np.int32)
        grid[0, :]  = WALL
        grid[-1, :] = WALL
        grid[:, 0]  = WALL
        grid[:, -1] = WALL
        return grid

    def _place_default_racks(self, grid: np.ndarray) -> np.ndarray:
        """Tempatkan rak dengan pola default berjajar rapi, sisakan lorong."""
        rh, rw    = self.cfg.rack_size
        inner_r   = self.cfg.rows - 4   # sisakan 2 baris atas (spawn) + 2 bawah (PD)
        inner_c   = self.cfg.cols - 2
        gap       = 2                   # lebar lorong antar rak
        start_row = 2                   # mulai dari row 2

        r = start_row
        while r + rh <= self.cfg.rows - 3:
            c = 1
            while c + rw <= self.cfg.cols - 1:
                # Cek apakah cukup ruang lorong
                can_place = True
                for dr in range(rh):
                    for dc in range(rw):
                        nr, nc = r + dr, c + dc
                        if grid[nr][nc] != FREE:
                            can_place = False
                if can_place:
                    for dr in range(rh):
                        for dc in range(rw):
                            grid[r + dr][c + dc] = RACK
                c += rw + gap
            r += rh + gap
        return grid

    def _place_random_racks(self, grid: np.ndarray) -> np.ndarray:
        """Tempatkan rak secara random dengan validasi."""
        rh, rw   = self.cfg.rack_size
        placed   = 0
        attempts = 0
        max_att  = 1000
        while placed < self.cfg.n_rack_blocks and attempts < max_att:
            r = random.randint(2, self.cfg.rows - 4)
            c = random.randint(1, self.cfg.cols - rw - 2)
            if self._is_valid_rack_pos(grid, r, c, rh, rw):
                for dr in range(rh):
                    for dc in range(rw):
                        grid[r + dr][c + dc] = RACK
                placed += 1
            attempts += 1
        if placed < self.cfg.n_rack_blocks:
            print(f"  ⚠ Hanya berhasil menempatkan {placed}/{self.cfg.n_rack_blocks} blok rak.")
        return grid

    def _place_manual_racks(self, grid: np.ndarray) -> np.ndarray:
        """User input koordinat rak satu per satu via CLI."""
        rh, rw = self.cfg.rack_size
        print(f"\n  Manual rack placement (ukuran blok {rh}x{rw})")
        print(f"  Area valid: row 2-{self.cfg.rows-3}, col 1-{self.cfg.cols-rw-1}")
        print("  Ketik 'selesai' untuk berhenti.\n")
        idx = 1
        while True:
            raw = input(f"  Blok rak {idx} — masukkan 'row,col' (atau 'selesai'): ").strip()
            if raw.lower() == "selesai":
                break
            try:
                r, c = map(int, raw.split(","))
                if self._is_valid_rack_pos(grid, r, c, rh, rw):
                    for dr in range(rh):
                        for dc in range(rw):
                            grid[r + dr][c + dc] = RACK
                    print(f"  ✅ Rak {idx} ditempatkan di ({r},{c})")
                    idx += 1
                else:
                    print("  ⚠ Posisi tidak valid atau overlap.")
            except (ValueError, IndexError):
                print("  ⚠ Format salah. Contoh: 3,4")
        return grid

    def _place_pickup_dropoff(self, grid: np.ndarray) -> np.ndarray:
        """Tempatkan pick-up dan drop-off di baris kedua dari bawah."""
        cfg   = self.cfg
        row   = cfg.rows - 2
        n     = cfg.n_forklifts
        step  = max(1, (cfg.cols - 2) // (n * 2))
        cfg.pickup_positions  = []
        cfg.dropoff_positions = []
        for i in range(n):
            pc = 1 + i * (step * 2)
            dc = pc + 1
            if dc < cfg.cols - 1:
                grid[row][pc] = PICKUP
                grid[row][dc] = DROPOFF
                cfg.pickup_positions.append((row, pc))
                cfg.dropoff_positions.append((row, dc))
        return grid

    def _place_forklifts(self, grid: np.ndarray) -> np.ndarray:
        """Tempatkan forklift spawn di posisi yang sudah ditentukan."""
        for pos in self.cfg.forklift_positions:
            r, c = pos
            if grid[r][c] == FREE:
                grid[r][c] = FORKLIFT
        return grid

    def _is_valid_rack_pos(self, grid, r, c, rh, rw) -> bool:
        """Cek posisi rak valid: tidak overlap wall/rak/PD/FK, bukan baris spawn atau PD."""
        cfg = self.cfg
        if r < 2 or r + rh > cfg.rows - 2:
            return False
        if c < 1 or c + rw > cfg.cols - 1:
            return False
        for dr in range(rh):
            for dc in range(rw):
                if grid[r + dr][c + dc] != FREE:
                    return False
        # Pastikan ada setidaknya 1 cell lorong di sekitar rak
        for dr in range(-1, rh + 1):
            for dc in range(-1, rw + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.rows and 0 <= nc < cfg.cols:
                    if grid[nr][nc] == WALL:
                        return False
        return True


# ─────────────────────────────────────────────
#  3. KELAS FORKLIFT
# ─────────────────────────────────────────────

class Forklift:
    def __init__(self, fid: int, spawn: tuple):
        self.id            = fid
        self.pos           = list(spawn)
        self.spawn_pos     = list(spawn)
        self.task          = None
        self.loaded        = False
        self.total_dist    = 0
        self.tasks_done    = 0
        self.collision_cnt = 0
        self.path_history  = [list(spawn)]  # riwayat jalur

    def reset(self):
        self.pos           = list(self.spawn_pos)
        self.task          = None
        self.loaded        = False
        self.total_dist    = 0
        self.tasks_done    = 0
        self.collision_cnt = 0
        self.path_history  = [list(self.spawn_pos)]

    def __repr__(self):
        return f"Forklift(id={self.id}, pos={self.pos}, loaded={self.loaded})"


# ─────────────────────────────────────────────
#  4. WAREHOUSE ENVIRONMENT (GYM)
# ─────────────────────────────────────────────

class WarehouseEnv(gym.Env):
    """
    Multi-Forklift Warehouse Gym Environment.
    Dibuat dari WarehouseConfig yang sudah dikonfigurasi user.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: WarehouseConfig, render_mode="human"):
        super().__init__()
        self.cfg          = cfg
        self.base_map     = deepcopy(cfg.grid)
        self.rows         = cfg.rows
        self.cols         = cfg.cols
        self.n_forklifts  = cfg.n_forklifts
        self.render_mode  = render_mode
        self.max_steps    = self.rows * self.cols * 2
        self.current_step = 0

        self.forklifts = [
            Forklift(i, cfg.forklift_positions[i])
            for i in range(self.n_forklifts)
        ]
        self.task_queue = []
        self._generate_tasks()

        # Spaces
        obs_size = 5 + (self.n_forklifts - 1) * 2
        self.observation_space = spaces.Dict({
            f"forklift_{i}": spaces.Box(
                low=np.zeros(obs_size, dtype=np.float32),
                high=np.array([self.rows, self.cols] * (obs_size // 2 + 1),
                              dtype=np.float32)[:obs_size],
                dtype=np.float32
            ) for i in range(self.n_forklifts)
        })
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * self.n_forklifts)

        # Render
        self.fig = None
        self.ax  = None
        self.ax_info = None

        # ── internal state placeholder (methods below) ──
        pass

    def _generate_tasks(self):
        self.task_queue = []
        for i in range(len(self.cfg.pickup_positions)):
            self.task_queue.append({
                "pickup" : self.cfg.pickup_positions[i],
                "dropoff": self.cfg.dropoff_positions[i],
                "done"   : False,
            })
        random.shuffle(self.task_queue)

    def _assign_tasks(self):
        available = [t for t in self.task_queue if not t["done"]]
        idx = 0
        for fk in self.forklifts:
            if fk.task is None and idx < len(available):
                fk.task   = available[idx]
                fk.loaded = False
                idx += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._generate_tasks()
        for fk in self.forklifts:
            fk.reset()
        self._assign_tasks()
        return self._get_observations(), self._get_info()

    def _get_observations(self) -> dict:
        obs = {}
        for i, fk in enumerate(self.forklifts):
            if fk.task is not None:
                target = fk.task["pickup"] if not fk.loaded else fk.task["dropoff"]
            else:
                target = fk.pos
            others = []
            for j, o in enumerate(self.forklifts):
                if j != i:
                    others.extend(o.pos)
            obs[f"forklift_{i}"] = np.array(
                [fk.pos[0], fk.pos[1], target[0], target[1], float(fk.loaded)] + others,
                dtype=np.float32
            )
        return obs

    def step(self, actions):
        self.current_step += 1
        rewards    = np.zeros(self.n_forklifts, dtype=np.float32)
        truncated  = self.current_step >= self.max_steps
        terminated = False

        new_pos = []
        for i, fk in enumerate(self.forklifts):
            dr, dc = DELTA[int(actions[i])]
            new_pos.append([fk.pos[0] + dr, fk.pos[1] + dc])

        for i, fk in enumerate(self.forklifts):
            nr, nc = new_pos[i]
            cell   = self.base_map[nr][nc]
            if cell in [WALL, RACK]:
                rewards[i]  -= 2.0
                new_pos[i]   = fk.pos[:]
                fk.collision_cnt += 1
                continue
            for j in range(self.n_forklifts):
                if i != j and new_pos[i] == new_pos[j]:
                    rewards[i] -= 5.0
                    new_pos[i]  = fk.pos[:]
                    fk.collision_cnt += 1
                    break

        rewards -= 0.1

        for i, fk in enumerate(self.forklifts):
            if new_pos[i] != fk.pos:
                fk.total_dist += 1
            fk.pos = new_pos[i][:]
            fk.path_history.append(fk.pos[:])

            if fk.task is None:
                continue
            pt = tuple(fk.pos)
            if not fk.loaded and pt == tuple(fk.task["pickup"]):
                fk.loaded = True;  rewards[i] += 5.0
            elif fk.loaded and pt == tuple(fk.task["dropoff"]):
                fk.loaded = False; fk.tasks_done += 1
                fk.task["done"] = True; fk.task = None
                rewards[i] += 10.0

        self._assign_tasks()
        if all(t["done"] for t in self.task_queue):
            terminated = True; rewards += 50.0

        return self._get_observations(), rewards, terminated, truncated, self._get_info()

    def _get_info(self) -> dict:
        return {
            "step"       : self.current_step,
            "tasks_done" : sum(t["done"] for t in self.task_queue),
            "tasks_total": len(self.task_queue),
            "forklift_pos": [fk.pos[:] for fk in self.forklifts],
            "total_dist" : [fk.total_dist for fk in self.forklifts],
            "collisions" : [fk.collision_cnt for fk in self.forklifts],
        }

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# ─────────────────────────────────────────────
#  5. ANIMATOR
# ─────────────────────────────────────────────

class WarehouseAnimator:
    """
    Animasi step-by-step pergerakan forklift dengan kontrol:
    ▶ Play  |  ⏸ Pause  |  ⏭ Step  |  🔄 Reset  |  Kecepatan slider
    """
    def __init__(self, env: WarehouseEnv):
        self.env      = env
        self.running  = False
        self.done     = False
        self.step_num = 0
        self.interval = 400
        self.anim     = None
        self.obs, _   = env.reset()
        self._build_figure()

    # ──────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
        self.fig.suptitle(
            "Warehouse Multi-Forklift — Animation",
            color="white", fontsize=13, fontweight="bold", y=0.98
        )
        gs = GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[2.5, 1],
            height_ratios=[10, 1],
            hspace=0.08, wspace=0.08
        )
        self.ax      = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_ctrl = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis("off")
        self.ax_ctrl.axis("off")
        self.fig.patch.set_facecolor("#0f1117")

        # ── Tombol kontrol ──
        btn_y = 0.02
        ax_play  = self.fig.add_axes([0.10, btn_y, 0.10, 0.045])
        ax_pause = self.fig.add_axes([0.21, btn_y, 0.10, 0.045])
        ax_step  = self.fig.add_axes([0.32, btn_y, 0.10, 0.045])
        ax_reset = self.fig.add_axes([0.43, btn_y, 0.10, 0.045])
        ax_fast  = self.fig.add_axes([0.60, btn_y, 0.08, 0.045])
        ax_slow  = self.fig.add_axes([0.69, btn_y, 0.08, 0.045])

        btn_style = {"color": "#1e2433", "hovercolor": "#374151"}
        self.btn_play  = Button(ax_play,  "▶  Play",  **btn_style)
        self.btn_pause = Button(ax_pause, "⏸  Pause", **btn_style)
        self.btn_step  = Button(ax_step,  "⏭  Step",  **btn_style)
        self.btn_reset = Button(ax_reset, "🔄 Reset", **btn_style)
        self.btn_fast  = Button(ax_fast,  "⏩ Fast",  **btn_style)
        self.btn_slow  = Button(ax_slow,  "⏪ Slow",  **btn_style)

        for btn in [self.btn_play, self.btn_pause, self.btn_step,
                    self.btn_reset, self.btn_fast, self.btn_slow]:
            btn.label.set_color("white")
            btn.label.set_fontsize(9)

        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_fast.on_clicked(self._on_fast)
        self.btn_slow.on_clicked(self._on_slow)

        self._draw_frame()

    # ──────────────────────────────────────────
    def _draw_frame(self):
        env   = self.env
        rows  = env.rows
        cols  = env.cols

        self.ax.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax.set_facecolor("#0f1117")

        # ── Gambar setiap cell ──
        for r in range(rows):
            for c in range(cols):
                val   = env.base_map[r][c]
                color = CELL_COLORS.get(val, "#1a2332")
                rect  = mpatches.FancyBboxPatch(
                    (c + 0.04, rows - r - 1 + 0.04), 0.92, 0.92,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="#2d3748", linewidth=0.5
                )
                self.ax.add_patch(rect)
                if val in CELL_LABELS:
                    lbl, lc = CELL_LABELS[val]
                    self.ax.text(
                        c + 0.5, rows - r - 0.5, lbl,
                        ha="center", va="center",
                        color=lc, fontsize=8, fontweight="bold"
                    )
                # Koordinat kecil
                self.ax.text(
                    c + 0.08, rows - r - 0.88,
                    f"{r},{c}", color="#334155", fontsize=4.5
                )

        # ── Gambar trail riwayat jalur ──
        for i, fk in enumerate(env.forklifts):
            col = FORKLIFT_COLORS[i % len(FORKLIFT_COLORS)]
            hist = fk.path_history
            if len(hist) > 1:
                xs = [h[1] + 0.5 for h in hist]
                ys = [rows - h[0] - 0.5 for h in hist]
                self.ax.plot(xs, ys, color=col, alpha=0.25,
                             linewidth=1.2, linestyle="--", zorder=3)

        # ── Gambar forklift ──
        for i, fk in enumerate(env.forklifts):
            r, c  = fk.pos
            col   = FORKLIFT_COLORS[i % len(FORKLIFT_COLORS)]
            circle = plt.Circle(
                (c + 0.5, rows - r - 0.5), 0.38,
                color=col, zorder=6
            )
            self.ax.add_patch(circle)
            label = f"L{i+1}" if fk.loaded else str(i + 1)
            self.ax.text(
                c + 0.5, rows - r - 0.5, label,
                ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=7
            )
            # Panah ke target
            if fk.task is not None:
                tgt  = fk.task["pickup"] if not fk.loaded else fk.task["dropoff"]
                tr, tc = tgt
                self.ax.annotate(
                    "", xy=(tc + 0.5, rows - tr - 0.5),
                    xytext=(c + 0.5, rows - r - 0.5),
                    arrowprops=dict(
                        arrowstyle="->", color=col,
                        lw=1.5, linestyle="dashed", alpha=0.7
                    ), zorder=5
                )

        # ── Styling axes ──
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_xticklabels(range(cols), color="#64748b", fontsize=7)
        self.ax.set_yticklabels(range(rows - 1, -1, -1), color="#64748b", fontsize=7)
        self.ax.grid(True, color="#2d3748", linewidth=0.3)
        info  = env._get_info()
        status = "✅ DONE!" if self.done else ("⏸" if not self.running else "▶")
        self.ax.set_title(
            f"{status}  Step: {self.step_num}  |  "
            f"Tasks: {info['tasks_done']}/{info['tasks_total']}  |  "
            f"Speed: {self.interval}ms/frame",
            color="white", fontsize=10, pad=8
        )
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2d3748")

        # ── Panel info kanan ──
        lines = ["FORKLIFT STATUS\n" + "─" * 26 + "\n"]
        for i, fk in enumerate(env.forklifts):
            lines.append(
                f"Forklift {i+1}\n"
                f"  Pos    : ({fk.pos[0]}, {fk.pos[1]})\n"
                f"  Status : {'Loaded ✓' if fk.loaded else 'Empty'}\n"
                f"  Tasks  : {fk.tasks_done} done\n"
                f"  Dist   : {fk.total_dist} steps\n"
                f"  Crash  : {fk.collision_cnt}\n"
            )
        lines.append("─" * 26 + "\n")
        lines.append(f"Total steps   : {self.step_num}\n")
        lines.append(f"Total dist    : {sum(fk.total_dist for fk in env.forklifts)}\n")
        lines.append(f"Total crashes : {sum(fk.collision_cnt for fk in env.forklifts)}\n")

        self.ax_info.text(
            0.05, 0.95, "".join(lines),
            transform=self.ax_info.transAxes,
            fontsize=8.5, verticalalignment="top",
            color="#e2e8f0", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#1e2433", edgecolor="#2d3748")
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color=CELL_COLORS[RACK],    label="Rack"),
            mpatches.Patch(color=CELL_COLORS[PICKUP],  label="Pick-up (P)"),
            mpatches.Patch(color=CELL_COLORS[DROPOFF], label="Drop-off (D)"),
        ] + [
            mpatches.Patch(color=FORKLIFT_COLORS[i], label=f"Forklift {i+1}")
            for i in range(env.n_forklifts)
        ]
        self.ax.legend(
            handles=legend_elements, loc="upper right",
            fontsize=7, facecolor="#1e2433",
            edgecolor="#2d3748", labelcolor="white", framealpha=0.9
        )

        self.fig.canvas.draw_idle()

    # ──────────────────────────────────────────
    def _animate(self, frame):
        if not self.running or self.done:
            return
        actions = self.env.action_space.sample()
        _, _, terminated, truncated, _ = self.env.step(actions)
        self.step_num += 1
        if terminated or truncated:
            self.done    = True
            self.running = False
        self._draw_frame()

    def _on_play(self, event):
        if self.done:
            return
        self.running = True
        if self.anim is None:
            self.anim = animation.FuncAnimation(
                self.fig, self._animate,
                interval=self.interval, cache_frame_data=False
            )
        self.fig.canvas.draw_idle()

    def _on_pause(self, event):
        self.running = False
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def _on_step(self, event):
        if self.done:
            return
        self.running = False
        actions = self.env.action_space.sample()
        _, _, terminated, truncated, _ = self.env.step(actions)
        self.step_num += 1
        if terminated or truncated:
            self.done = True
        self._draw_frame()

    def _on_reset(self, event):
        self.running  = False
        self.done     = False
        self.step_num = 0
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        self.obs, _ = self.env.reset()
        self._draw_frame()

    def _on_fast(self, event):
        self.interval = max(50, self.interval - 100)
        if self.anim is not None:
            self.anim.event_source.interval = self.interval
        self._draw_frame()

    def _on_slow(self, event):
        self.interval = min(2000, self.interval + 200)
        if self.anim is not None:
            self.anim.event_source.interval = self.interval
        self._draw_frame()

    def start(self):
        plt.show()


# ─────────────────────────────────────────────
#  6. MAIN
# ─────────────────────────────────────────────

def main():
    cfg = input_wizard()
    gen = MapGenerator(cfg)
    cfg.grid = gen.generate()
    print_map(cfg)
    env = WarehouseEnv(cfg, render_mode="human")
    animator = WarehouseAnimator(env)
    animator.start()


def print_map(cfg: WarehouseConfig):
    """Print map ke terminal sebagai teks."""
    symbols = {FREE:"·", WALL:"█", RACK:"▣", PICKUP:"P",
               DROPOFF:"D", FORKLIFT:"F", CHARGING:"⚡"}
    print("\n  Generated Map:")
    for r in range(cfg.rows):
        row_str = "  "
        for c in range(cfg.cols):
            row_str += symbols.get(cfg.grid[r][c], "?") + " "
        print(row_str)
    print()


if __name__ == "__main__":
    main()