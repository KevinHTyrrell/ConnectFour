import collections
import numpy as np
from Grid import Grid


class Game:
    def __init__(self):
        self._player_one    = 1
        self._player_two    = 2
        self._connection    = 4
        self._grid          = Grid()
        self._run_game      = True

    def _check_arr(self, selected_arr):
        piece_count_dict = collections.Counter(selected_arr)
        for piece in [self._player_one, self._player_two]:
            piece_count = piece_count_dict.get(piece)
            if piece_count is None:
                continue
            if piece_count >= self._connection:
                split_row = np.split(selected_arr, np.where(np.diff(selected_arr) != 0)[0] + 1)
                count_result = [len(x) >= self._connection for x in split_row]
                if any(count_result):
                    return True
        return False

    def _check_cols(self):
        n_cols = self._grid.get_width()
        for col in range(n_cols):
            selected_col = self._grid.get_grid()[:, col]
            if self._check_arr(selected_col):
                return True
        return False

    def _check_rows(self):
        n_rows = self._grid.get_height()
        for row in range(n_rows):
            selected_row = self._grid.get_grid()[row, :]
            if self._check_arr(selected_row):
                return True
        return False

    def _check_diag(self):
        for row in range(self._grid.get_height() - self._connection):
            for col in range(self._grid.get_width() - self._connection):
                selected_grid = self._grid.get_grid()[row:(row+self._connection), col:(col+self._connection)]
                grid_result = [self._check_arr(np.diag(grd)) for grd in [selected_grid, selected_grid[:, ::-1]]]
                if any(grid_result):
                    return True
        return False

    def _check_win(self):
        row_win = self._check_rows()
        col_win = self._check_cols()
        diag_win = self._check_diag()
        if any([row_win, col_win, diag_win]):
            return True
        return False

    def _end_game(self):
        self._run_game = False

    def _get_player_input(self, player: int):
        print(f'Player {player} enter column?:', end='', flush=True)
        insert_col = input()
        valid_move = self._validate_input(insert_col)
        if valid_move:
            insert_col_int = int(insert_col)
            return insert_col_int
        else:
            return None

    def _validate_input(self, user_input):
        if user_input.isnumeric():
            user_input = int(user_input)
        else:
            return False
        if user_input >= self._grid.get_width():
            return False
        if user_input < 0:
            return False
        return self._grid.check_valid_move(col=user_input)

    def run_game(self):
        n_iterations = 0
        while self._run_game:
            player = (n_iterations % 2) + 1
            self._grid.print()
            col_to_insert = self._get_player_input(player)
            if col_to_insert is None:
                print('INVALID MOVE')
                continue
            self._grid.insert_piece(piece=player, col=col_to_insert)
            n_iterations += 1
            is_win = self._check_win()
            if is_win:
                print(f'\n\n\n\n\n\n\nPLAYER {player} WINS')
                self._end_game()
