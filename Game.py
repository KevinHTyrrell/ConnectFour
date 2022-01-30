import collections
from Grid import Grid


class Game:
    def __init__(self):
        self._player_one    = 1
        self._player_two    = 2
        self._connection    = 4
        self._grid          = Grid()

    def player_one_move(self, col):
        if col >= self._grid.get_width():
            print('INVALID MOVE')
            return False
        else:
            self._grid.insert_piece(self._player_one, col)
            return True

    def player_two_move(self, col):
        if col >= self._grid.get_width():
            print('INVALID MOVE')
            return False
        else:
            self._grid.insert_piece(self._player_two, col)
            return True

    def check_for_win(self):
        grid = self._grid.get_grid()
        n_rows = self._grid.get_height()
        n_cols = self._grid.get_width()
        self._check_rows()
        self._check_cols()
        # scan diagonal

    def _check_rows(self):
        n_rows = self._grid.get_height()
        for row in range(n_rows):
            selected_row = self._grid.get_grid()[row, :]
            is_win = self._check_arr(selected_row)

    def _check_cols(self):
        n_cols = self._grid.get_width()
        for col in range(n_cols):
            selected_col = self._grid.get_grid()[:, col]
            is_win = self._check_arr(selected_col)

    def _check_arr(self, selected_arr):
        piece_count = collections.Counter(selected_arr)
        for piece in [self._player_one, self._player_two]:
            if piece_count.get(piece) >= self._connection:
                split_row = np.split(selected_arr, np.where(np.diff(selected_arr) != 0)[0] + 1)
                count_result = [len(x) >= self._connection for x in split_row]
                if any(count_result):
                    print(f'PLAYER {piece} WINS')
                    return True
        return False