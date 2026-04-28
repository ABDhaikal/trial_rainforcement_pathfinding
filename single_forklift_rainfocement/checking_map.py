



from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


selected_map_data = Path(__file__).with_name("warehouse_dataset.npz")
selected_data_index = 10000


def main() -> None:
	if not selected_map_data.exists():
		raise FileNotFoundError(f"Map file not found: {selected_map_data}")

	data = np.load(selected_map_data, allow_pickle=True)

	print(f"Loaded: {selected_map_data}")
	print(f"Available keys: {list(data.keys())}")

	if len(data.files) == 0:
		print("No data found in the file.")
		return

	key = data.files[0]
	values = data[key]

	print(f"Using key: {key}")
	print(f"Data shape: {values.shape}")

	if values.ndim == 0:
		print(f"Value: {values.item()}")
		return

	if selected_data_index < 0 or selected_data_index >= len(values):
		raise IndexError(
			f"selected_data_index out of range: {selected_data_index} (size={len(values)})"
		)

	selected_sample = values[selected_data_index]
	print(f"Selected index: {selected_data_index}")
	print("Selected sample:")
	print(selected_sample)

	# Use reversed grayscale so that 0 -> white and 1 -> black
	plt.imshow(selected_sample, cmap="gray_r", vmin=0, vmax=1)
	plt.title(f"Map Sample (Index: {selected_data_index})")
	plt.colorbar(label="Cell Type (0=Empty, 1=Rack)")
	plt.show()


if __name__ == "__main__":
	main()
