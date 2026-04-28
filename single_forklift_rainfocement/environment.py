import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

class PathFindingEnv(gym.Env):
    """
    Environment Reinforcement Learning untuk Navigasi Warehouse.
    Menggunakan dataset peta statis yang sudah dipastikan solvable (memiliki jalan).
    Ditambah memori 'Aksi Terakhir' sebagai input State.
    """
    def __init__(self, dataset_path="warehouse_dataset.npz", max_steps=200):
        super(PathFindingEnv, self).__init__()
        
        # 1. Memuat Dataset Peta (Dibuat oleh Program 1)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File dataset {dataset_path} tidak ditemukan! Jalankan generator peta terlebih dahulu.")
        
        dataset = np.load(dataset_path)
        self.all_grids = dataset['grids']
        self.all_agents_start = dataset['agents']
        self.all_goals = dataset['goals']
        self.num_maps = len(self.all_grids)
        
        # Ambil dimensi peta dari data pertama
        self.height, self.width = self.all_grids[0].shape
        
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = 4 # Default 4 (Diam)
        
        # 2. Mendefinisikan Ruang Aksi (Action Space)
        # Sesuai permintaan: Kanan, Kiri, Atas, Bawah, Diam (Total 5 aksi)
        # 0: Kanan, 1: Kiri, 2: Atas, 3: Bawah, 4: Diam
        self.action_space = spaces.Discrete(5)
        
        # 3. Mendefinisikan Ruang Observasi (Observation Space / FNN Input)
        # Input = Peta diratakan (width * height) 
        #         + 4 (Koordinat Normalisasi agen_y, agen_x, goal_y, goal_x)
        #         + 5 (One-Hot Encoding untuk memori aksi sebelumnya)
        self.obs_size = (self.width * self.height) + 4 + 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Memilih peta baru dari dataset pada setiap awal episode."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Set aksi terakhir ke 'Diam' setiap episode baru mulai
        self.last_action = 4 
        
        # Pilih satu indeks peta secara acak dari dataset
        idx = self.np_random.integers(0, self.num_maps)
        
     # Muat grid peta, posisi awal agen, dan posisi goal
        self.grid = self.all_grids[idx].copy()
        
        # Konversi uint16 dari NumPy menjadi integer standar Python 
        # agar bisa dikalkulasi dengan angka negatif (dx/dy = -1)
        self.agent_pos = [int(self.all_agents_start[idx][0]), int(self.all_agents_start[idx][1])]
        self.goal_pos = [int(self.all_goals[idx][0]), int(self.all_goals[idx][1])]
        
        return self._get_state(), {}

    def step(self, action):
        """Mengeksekusi langkah dari Neural Network dan menghitung Reward."""
        self.current_step += 1
        
        # --- PENANGANAN INPUT ARRAY 1x5 ---
        # Jika action yang masuk berupa array probabilitas dari NN, ambil index terbesarnya
        if isinstance(action, (np.ndarray, list)):
            action = int(np.argmax(action))
        else:
            action = int(action)

        # Simpan aksi ini ke memori untuk dibaca di step berikutnya
        self.last_action = action

        # Pemetaan aksi ke pergerakan koordinat [delta_Y, delta_X]
        # 0: Kanan, 1: Kiri, 2: Atas, 3: Bawah, 4: Diam
        gerak = {
            0: [0, 1],
            1: [0, -1],
            2: [-1, 0],
            3: [1, 0],
            4: [0, 0]
        }
        
        dy, dx = gerak[action]
        
        # Cek jika aksi adalah Diam
        if action == 4:
            reward = -2
            terminated = False
            
            # Cek kehabisan langkah meskipun agen diam
            truncated = self.current_step >= self.max_steps
            return self._get_state(), reward, terminated, truncated, {}

        # --- HITUNG POSISI BARU JIKA BERGERAK ---
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx
        
        # Deteksi Tabrakan: Batas Peta atau Nabrak Rak (Nilai grid == 1)
        if (new_y < 0 or new_y >= self.height or 
            new_x < 0 or new_x >= self.width or 
            self.grid[new_y, new_x] == 1):
            
            # Jika menabrak
            reward = -100
            terminated = True # Episode langsung berhenti (Gagal)
            truncated = False
            # Agen tidak pindah posisi jika menabrak tembok
            
        else:
            # Jika pergerakan aman, update posisi agen
            self.agent_pos = [new_y, new_x]
            
            # Cek apakah mencapai Goal
            if self.agent_pos == self.goal_pos:
                reward = 100
                terminated = True # Episode langsung berhenti (Sukses)
                truncated = False
            else:
                # Jika sekadar bergerak aman di lantai kosong
                reward = -1
                terminated = False
                truncated = False

        # Hentikan paksa jika melebihi batas maksimal langkah (mencegah loop tak terhingga)
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_state(), float(reward), terminated, truncated, {}

    def _get_state(self):
        """Merakit state/observasi ke dalam Array 1D untuk Feedforward Neural Network."""
        # 1. Peta grid diratakan (flatten)
        flattened_grid = self.grid.flatten().astype(np.float32)
        
        # 2. Normalisasi koordinat agen dan goal (dari 0 hingga 1)
        norm_ay = self.agent_pos[0] / (self.height - 1)
        norm_ax = self.agent_pos[1] / (self.width - 1)
        norm_gy = self.goal_pos[0] / (self.height - 1)
        norm_gx = self.goal_pos[1] / (self.width - 1)
        
        coords = np.array([norm_ay, norm_ax, norm_gy, norm_gx], dtype=np.float32)
        
        # 3. Memori Aksi Terakhir (One-Hot Encoding 5 aksi)
        action_array = np.zeros(5, dtype=np.float32)
        action_array[self.last_action] = 1.0
        
        # 4. Gabungkan peta, koordinat, dan aksi terakhir
        state = np.concatenate((flattened_grid, coords, action_array))
        return state

    def render(self):
        """Visualisasi arena latihan."""
        display_grid = self.grid.copy()
        
        # Tandai posisi agen dan goal di grid untuk digambar
        display_grid[self.agent_pos[0], self.agent_pos[1]] = 2
        display_grid[self.goal_pos[0], self.goal_pos[1]] = 3
        
        cmap = ListedColormap(["#f5f5f5", "#2f4f4f", "#3498db", "#e74c3c"])
        plt.figure(figsize=(5, 5))
        plt.imshow(display_grid, cmap=cmap)
        plt.title(f"Step: {self.current_step} | Posisi: {self.agent_pos} | Aksi Terakhir: {self.last_action}")
        plt.xticks(np.arange(-0.5, self.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, self.height, 1), minor=True)
        plt.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
        plt.axis('on')
        
        # Tambahkan delay sedikit jika merender dalam loop animasi
        plt.pause(0.1) 
        plt.clf()

# --- BLOK PENGUJIAN ENVIRONMENT ---
if __name__ == "__main__":
    # Pastikan Anda telah menjalankan program generator dataset sebelumnya!
    try:
        env = PathFindingEnv(dataset_path="warehouse_dataset.npz")
        
        print("Mereset Environment...")
        state, info = env.reset()
        print(f"Ukuran State Input ke NN: {state.shape} neuron")
        print(f"Step Awal: {env.current_step}")
        
        print("\nMenguji Langkah Pertama (Memilih aksi Diam)...")
        # Misal NN mengeluarkan output tertinggi di index 4 (Diam)
        action_array_1 = [0.0, 0.0, 0.0, 0.0, 1.0] 
        next_state, reward, terminated, truncated, info = env.step(action_array_1)
        print(f"Reward: {reward}")
        print(f"5 Nilai Terakhir pada State (One-Hot Action): {next_state[-5:]}")
        
        print("\nMenguji Langkah Kedua (Memilih aksi Kiri)...")
        # Misal NN mengeluarkan output tertinggi di index 1 (Kiri)
        action_array_2 = [0.1, 0.8, 0.0, 0.0, 0.1] 
        next_state, reward, terminated, truncated, info = env.step(action_array_2)
        print(f"Reward: {reward}")
        print(f"5 Nilai Terakhir pada State (One-Hot Action): {next_state[-5:]}")
        
         
        # tampilan peta dengan plotting biasa pada window terpisah
        print("\nMenampilkan Peta dengan Posisi Agen dan Goal...")
        display_grid = env.grid.copy()
        display_grid[env.agent_pos[0], env.agent_pos[1]] = 2
        display_grid[env.goal_pos[0], env.goal_pos[1]] = 3

        cmap = ListedColormap(["#f5f5f5", "#2f4f4f", "#3498db", "#e74c3c"])
        plt.figure(figsize=(5, 5))
        plt.imshow(display_grid, cmap=cmap)
        plt.title(f"Step: {env.current_step} | Posisi: {env.agent_pos} | Aksi Terakhir: {env.last_action}")
        plt.xticks(np.arange(-0.5, env.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, env.height, 1), minor=True)
        plt.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
        plt.axis('on')
        plt.show()
        
        env.close()
        
    except FileNotFoundError as e:
        print(e)