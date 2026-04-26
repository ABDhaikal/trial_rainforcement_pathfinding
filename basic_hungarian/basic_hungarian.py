from __future__ import annotations

from typing import List, Sequence, Tuple


def hungarian_method(cost_matrix: Sequence[Sequence[float]]) -> Tuple[List[Tuple[int, int]], float]:
	"""Selesaikan assignment minimum cost dengan Hungarian method.

	Args:
		cost_matrix: Matriks biaya berukuran m x n.

	Returns:
		Tuple (assignments, total_cost)
		- assignments: daftar pasangan (row_index, col_index)
		- total_cost: total biaya minimum dari assignment
	"""
	if not cost_matrix or not cost_matrix[0]:
		raise ValueError("cost_matrix tidak boleh kosong")

	row_count = len(cost_matrix)
	col_count = len(cost_matrix[0])

	for row in cost_matrix:
		if len(row) != col_count:
			raise ValueError("Semua baris pada cost_matrix harus memiliki panjang yang sama")

	# Hungarian O(n^3) dengan pendekatan potensial (u, v).
	# Implementasi inti memerlukan rows <= cols, jadi jika tidak, kita transpose.
	transposed = False
	matrix = [list(map(float, row)) for row in cost_matrix]
	if row_count > col_count:
		transposed = True
		matrix = [list(col) for col in zip(*matrix)]
		row_count, col_count = col_count, row_count

	u = [0.0] * (row_count + 1)
	v = [0.0] * (col_count + 1)
	p = [0] * (col_count + 1)
	way = [0] * (col_count + 1)

	for i in range(1, row_count + 1):
		p[0] = i
		minv = [float("inf")] * (col_count + 1)
		used = [False] * (col_count + 1)
		j0 = 0

		while True:
			used[j0] = True
			i0 = p[j0]
			delta = float("inf")
			j1 = 0

			for j in range(1, col_count + 1):
				if used[j]:
					continue

				cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
				if cur < minv[j]:
					minv[j] = cur
					way[j] = j0

				if minv[j] < delta:
					delta = minv[j]
					j1 = j

			for j in range(col_count + 1):
				if used[j]:
					u[p[j]] += delta
					v[j] -= delta
				else:
					minv[j] -= delta

			j0 = j1
			if p[j0] == 0:
				break

		while True:
			j1 = way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0:
				break

	assignment_by_row = [-1] * row_count
	for j in range(1, col_count + 1):
		if p[j] != 0:
			assignment_by_row[p[j] - 1] = j - 1

	assignments: List[Tuple[int, int]] = []
	total_cost = 0.0
	for r, c in enumerate(assignment_by_row):
		if c == -1:
			continue

		if transposed:
			# Karena transpose, indeks harus dikembalikan ke orientasi asli.
			original_row, original_col = c, r
		else:
			original_row, original_col = r, c

		assignments.append((original_row, original_col))
		total_cost += float(cost_matrix[original_row][original_col])

	return assignments, total_cost


if __name__ == "__main__":
	# Contoh: baris = forklift, kolom = tujuan/pekerjaan, nilai = biaya/jarak.
	matrix = [
		[4, 1, 3],
		[2, 0, 5],
		[3, 2, 2],
	]

	result, cost = hungarian_method(matrix)

	print("Cost matrix:")
	for row in matrix:
		print(row)

	print("\nAssignment optimal (forklift -> goal):")
	for forklift_id, goal_id in result:
		print(f"Forklift {forklift_id} -> Goal {goal_id} (biaya={matrix[forklift_id][goal_id]})")

	print(f"\nTotal biaya minimum: {cost}")
