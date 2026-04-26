
import torch

import csv
import numpy as np

def export_model(model, model_path="trained_dqn_model.pth"):
    torch.save(model.state_dict(), model_path)
    print(f"Model berhasil diekspor ke: {model_path}")


def export_qtable(model, replay_memory, qtable_path="qtable_export.csv"):
    # Q-table diekspor dari state unik yang pernah dikunjungi selama training.
    unique_states = {}
    for state, _, _, _, _ in replay_memory:
        state_np = state.numpy()
        state_key = tuple(np.round(state_np, 4))
        if state_key not in unique_states:
            unique_states[state_key] = state_np

    if not unique_states:
        print("Replay memory kosong, Q-table tidak diekspor.")
        return

    state_matrix = torch.FloatTensor(np.array(list(unique_states.values())))
    with torch.no_grad():
        q_values = model(state_matrix).numpy()

    with open(qtable_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"state_{i + 1}" for i in range(12)] + ["q_action_0", "q_action_1", "q_action_2", "best_action"]
        writer.writerow(header)

        for state_np, q_row in zip(state_matrix.numpy(), q_values):
            best_action = int(np.argmax(q_row))
            writer.writerow(list(state_np) + list(q_row) + [best_action])

    print(f"Q-table berhasil diekspor ke: {qtable_path}")


if __name__ == "__main__":
    print("Script export_model.py siap digunakan. Pastikan untuk menjalankan training terlebih dahulu untuk menghasilkan model dan replay memory yang valid sebelum mengekspor.")