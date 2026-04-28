
import numpy as np
import torch

from rainhung import DQN
from environment import hungarian_rainforcement_environment as plant


def test_trained_model(cost_matrix, model_path="trained_dqn_model.pth"):
  env = plant()
  env.matrix = np.array(cost_matrix, dtype=np.int32)
  env.current_row = 0
  env.available_cols = [0, 1, 2]

  checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
  input_dim = checkpoint["network.0.weight"].shape[1]
  model = DQN(input_dim=input_dim, output_dim=3)
  model.load_state_dict(checkpoint)
  model.eval()

  state = torch.FloatTensor(env._get_state())
  done = False
  row_idx = 0
  assignments = []
  total_cost = 0

  while not done:
    with torch.no_grad():
      q_values = model(state)
      valid_actions = env.available_cols
      valid_q = [q_values[a].item() if a in valid_actions else -float("inf") for a in range(3)]
      action = int(np.argmax(valid_q))

    assignments.append((row_idx, action))
    total_cost += int(env.matrix[row_idx, action])

    next_state_np, _, done = env.step(action)
    state = torch.FloatTensor(next_state_np)
    row_idx += 1

  print("=== TEST MODEL TERLATIH ===")
  print("Cost Matrix:")
  print(env.matrix)
  print("\nAssignments:")
  for r, c in assignments:
    print(f"Baris {r} -> Kolom {c} (Cost: {env.matrix[r, c]})")
  print(f"\nTotal Cost: {total_cost}")


if __name__ == "__main__":
  matrix = [
    [4, 2, 3],
    [2, 3, 1],
    [3, 1, 2],
  ]
  test_trained_model(matrix)

