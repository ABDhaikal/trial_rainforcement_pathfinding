from collections import deque
import numpy as np

class hungarian_rainforcement_environment:
    def __init__(self):
        self.matrix = np.zeros((3, 3))
        self.reset()

    def reset(self):
        # Menggunakan rentang 0 hingga 10000 sesuai spesifikasi
        self.matrix = np.random.randint(0, 10001, size=(3, 3))
        self.current_row = 0
        self.available_cols = [0, 1, 2]
        return self._get_state()

    def _get_state(self):
        # Normalisasi matriks ke rentang 0-1 untuk stabilitas Neural Network
        flat_matrix = self.matrix.flatten() / 10000.0
        
        # Masking kolom yang tersedia (1 = tersedia, 0 = sudah diambil)
        mask = np.zeros(3)
        for c in self.available_cols:
            mask[c] = 1.0
            
        # State adalah gabungan informasi matriks dan kolom yang masih bisa dipilih
        return np.concatenate([flat_matrix, mask])

    def step(self, action):
        # Penalti jika memilih kolom yang sudah ditugaskan sebelumnya
        if action not in self.available_cols:
            return self._get_state(), -10.0, True  

        # Reward = -Cost (karena Hungarian Method mencari nilai minimum)
        # Dinormalisasi agar konvergensi lebih cepat
        cost = self.matrix[self.current_row, action]
        reward = -cost / 10000.0

        # Transisi State
        self.available_cols.remove(action)
        self.current_row += 1

        # Episode selesai jika 3 baris sudah mendapat tugas
        done = self.current_row == 3
        return self._get_state(), reward, done
    

if __name__ == "__main__":
    env = hungarian_rainforcement_environment()
    state = env.reset()
    print("Initial State:", state)

    done = False
    while not done:
        action = np.random.choice(env.available_cols)  # Pilih aksi secara acak dari kolom yang tersedia
        next_state, reward, done = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
