import numpy as np


class Grid:
    def __init__(self, height: int = 6, width: int = 7):
        self._height    = height
        self._width     = width
        self._grid      = self._create_grid()

    def _create_grid(self):
        grid_flat = np.zeros(self._width * self._height)
        grid_reshaped = np.reshape(grid_flat, newshape=(self._height, self._width))
        return grid_reshaped

    def print(self):
        print('\n\n\n\n  0  1  2  3  4  5  6')
        print('------------------------')
        print(self._grid[::-1])

    def insert_piece(self, piece: int, col: int):
        selected_column = self._grid[:, col]
        for row in range(selected_column.__len__()):
            if selected_column[row] == 0:
                selected_column[row] = piece
                break

    def check_valid_move(self, col: int):
        selected_column = self._grid[:, col]
        for row in range(selected_column.__len__()):
            if selected_column[row] == 0:
                return True
        return False

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_grid(self):
        return self._grid