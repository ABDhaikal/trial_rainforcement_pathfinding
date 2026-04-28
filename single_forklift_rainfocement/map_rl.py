import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class WarehouseDatasetGenerator:
    def __init__(self, width=10, height=10, rack_count=25, num_maps=1000, save_path="warehouse_dataset.npz"):
        self.width = width
        self.height = height
        self.rack_count = rack_count
        self.num_maps = num_maps # Jumlah peta yang ingin dihasilkan
        self.save_path = save_path
        
    def _is_solvable(self, grid, start, goal):
        """Algoritma BFS sebagai wasit untuk memastikan ada jalan menuju goal."""
        queue = deque([start])
        visited = set([start])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Atas, Bawah, Kiri, Kanan
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
                
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Cek batas peta dan pastikan bukan rak (grid value == 0)
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def _generate_single_valid_map(self):
        """Membuat satu peta, meletakkan agen & goal, dan memastikan solvable."""
        valid = False
        while not valid:
            grid = np.zeros((self.height, self.width), dtype=np.uint8)
            placed = 0
            
            # 1. Letakkan 25 rak secara acak
            # Catatan: Karena ukuran 10x10 cukup sempit untuk 25 rak (25% area tertutup),
            # proses pencarian peta yang valid mungkin memerlukan beberapa iterasi di while loop ini.
            while placed < self.rack_count:
                r = random.randint(0, self.height - 1)
                c = random.randint(0, self.width - 1)
                if grid[r, c] == 0:
                    grid[r, c] = 1
                    placed += 1
            
            # 2. Cari sel kosong untuk agen dan goal
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) < 2:
                continue # Peta terlalu penuh (sangat jarang terjadi untuk 25 rak)
                
            indices = np.random.choice(len(empty_cells), 2, replace=False)
            agent_pos = tuple(empty_cells[indices[0]])
            goal_pos = tuple(empty_cells[indices[1]])
            
            # 3. Validasi dengan BFS
            if self._is_solvable(grid, agent_pos, goal_pos):
                valid = True
                return grid, agent_pos, goal_pos

    def generate_dataset(self):
        """Fungsi utama untuk membuat dan menyimpan ribuan peta."""
        print(f"Mulai membuat {self.num_maps} peta berukuran {self.width}x{self.height} dengan {self.rack_count} rak...")
        
        all_grids = []
        all_agents = []
        all_goals = []
        
        for i in range(self.num_maps):
            grid, agent, goal = self._generate_single_valid_map()
            all_grids.append(grid)
            all_agents.append(agent)
            all_goals.append(goal)
            
            # Cetak progress setiap kelipatan 100 agar user tahu program tidak hang
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{self.num_maps}] Peta berhasil dibuat...")

        # Konversi ke array NumPy
        grids_np = np.array(all_grids, dtype=np.uint8)
        agents_np = np.array(all_agents, dtype=np.uint16)
        goals_np = np.array(all_goals, dtype=np.uint16)
        
        # Simpan ke dalam file terkompresi .npz (Sangat hemat ukuran)
        np.savez_compressed(
            self.save_path, 
            grids=grids_np, 
            agents=agents_np, 
            goals=goals_np
        )
        print(f"\nSelesai! Dataset tersimpan di: {os.path.abspath(self.save_path)}")
        print(f"Ukuran Array Grids: {grids_np.shape} (Jumlah Peta, Tinggi, Lebar)")

    def visualize_random_sample(self):
        """Memuat dataset yang sudah disimpan dan menampilkan 1 peta acak."""
        if not os.path.exists(self.save_path):
            print(f"File {self.save_path} tidak ditemukan!")
            return
            
        # Memuat file NPZ
        data = np.load(self.save_path)
        grids = data['grids']
        agents = data['agents']
        goals = data['goals']
        
        # Pilih satu indeks acak
        idx = random.randint(0, len(grids) - 1)
        grid_sample = grids[idx].copy()
        agent_pos = tuple(agents[idx])
        goal_pos = tuple(goals[idx])
        
        # Penandaan untuk visualisasi
        grid_sample[agent_pos] = 2
        grid_sample[goal_pos] = 3
        
        cmap = ListedColormap(["#f5f5f5", "#2f4f4f", "#3498db", "#e74c3c"])
        plt.figure(figsize=(6, 6))
        plt.imshow(grid_sample, cmap=cmap)
        plt.title(f"Sampel Peta ke-{idx+1} dari Dataset\n(Terjamin Solvable)")
        plt.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
        plt.xticks(np.arange(-0.5, self.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, self.height, 1), minor=True)
        # Menambahkan legenda sederhana
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#f5f5f5', edgecolor='gray', label='Lantai'),
            Patch(facecolor='#2f4f4f', edgecolor='gray', label='Rak'),
            Patch(facecolor='#3498db', edgecolor='gray', label='Agent Start'),
            Patch(facecolor='#e74c3c', edgecolor='gray', label='Goal')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1))
        plt.tight_layout()
        plt.show()

# --- EKSEKUSI PROGRAM ---
if __name__ == "__main__":
    # Inisialisasi Generator: 10x10 peta, 25 rak, membuat 1000 peta untuk dataset
    generator = WarehouseDatasetGenerator(width=10, height=10, rack_count=25, num_maps=100000)
    
    # 1. Jalankan proses pembuatan dan penyimpanan (Hanya butuh dijalankan sekali)
    generator.generate_dataset()
    
    # 2. Opsional: Tampilkan satu sampel untuk verifikasi visual
    generator.visualize_random_sample()