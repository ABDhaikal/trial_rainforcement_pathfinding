import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from export_model import export_model, export_qtable
from environment import hungarian_rainforcement_environment as plant

# Arsitektur Deep Q-Network (Controller)
class DQN(nn.Module):
    def __init__(self, input_dim=12, output_dim=3):
        super(DQN, self).__init__()
        # State vektor berukuran 12 (9 elemen matriks + 3 elemen mask)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 3. Proses Training
def train_agent():
    env = plant()
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Hyperparameters
    episodes = 3000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64
    memory = deque(maxlen=2000)

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        done = False
        
        while not done:

            # Epsilon-Greedy Policy untuk eksplorasi
            if random.random() < epsilon:
                # Eksplorasi tetap dibatasi pada aksi legal agar transisi tetap valid
                action = random.choice(env.available_cols)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    # Filter aksi ilegal (hanya pilih dari kolom yang tersedia)
                    valid_actions = env.available_cols
                    valid_q = [q_values[a].item() if a in valid_actions else -float('inf') for a in range(3)]
                    action = np.argmax(valid_q)
            
            next_state_np, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state_np)
            
            # Simpan transisi ke replay memory
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            # Experience Replay
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                s_batch, a_batch, r_batch, s_next_batch, d_batch = zip(*batch)
                
                s_batch = torch.stack(s_batch)
                a_batch = torch.LongTensor(a_batch).unsqueeze(1)
                r_batch = torch.FloatTensor(r_batch)
                s_next_batch = torch.stack(s_next_batch)
                d_batch = torch.FloatTensor(d_batch)
                
                # Persamaan Bellman: Q(s,a) = r + gamma * max(Q(s', a'))
                current_q = model(s_batch).gather(1, a_batch).squeeze()
                with torch.no_grad():
                    next_q_values = model(s_next_batch)
                    # Mask legal action disimpan pada 3 elemen terakhir state
                    next_mask = s_next_batch[:, -3:]
                    masked_next_q = next_q_values.masked_fill(next_mask == 0, -1e9)
                    max_next_q = masked_next_q.max(1)[0]
                    # Jika state terminal (semua kolom habis), paksa nilai bootstrap = 0
                    no_valid_action = next_mask.sum(dim=1) == 0
                    max_next_q = torch.where(no_valid_action, torch.zeros_like(max_next_q), max_next_q)
                    target_q = r_batch + (1 - d_batch) * gamma * max_next_q
                
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Peluruhan tingkat eksplorasi
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{episodes} Selesai.")
            
    return model, list(memory)


# 4. Evaluasi & Output
def test_agent(model):
    env = plant()
    state = env.reset()
    state = torch.FloatTensor(state)
    done = False
    
    assignments = []
    total_cost = 0
    row_idx = 0
    
    # Matriks uji
    test_matrix = env.matrix.copy()
    
    while not done:
        with torch.no_grad():
            q_values = model(state)
            # Masking kolom yang sudah diambil agar agen dipaksa taat aturan di tahap inference
            valid_actions = env.available_cols
            valid_q = [q_values[a].item() if a in valid_actions else -float('inf') for a in range(3)]
            action = np.argmax(valid_q)
            
        assignments.append((row_idx, action))
        total_cost += test_matrix[row_idx, action]
        
        next_state_np, _, done = env.step(action)
        state = torch.FloatTensor(next_state_np)
        row_idx += 1

    print("\n=== HASIL INFERENSI ===")
    print("Cost Matrix (3x3):")
    print(test_matrix)
    print("\nOutput (3 Assignment):")
    for r, c in assignments:
        print(f"-> Pekerja/Baris {r} ditugaskan ke Tugas/Kolom {c} (Cost: {test_matrix[r, c]})")
    print(f"Total Cost Optimal (Prediksi RL): {total_cost}")

# Menjalankan script
if __name__ == "__main__":
    print("Mulai proses training...")
    trained_model, replay_memory = train_agent()
    export_model(trained_model)
    export_qtable(trained_model, replay_memory)
    test_agent(trained_model)