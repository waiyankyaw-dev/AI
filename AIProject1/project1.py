import numpy as np
import random
import time

try:
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

DIRECTIONS_NUMPY = np.array(DIRECTIONS, dtype=np.int8)

POSITION_WEIGHT = np.array([
    [8,-3, 2, 2, 2, 2, -3, 8],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 2, -1, 0, 0, 0, 0, -1, 2],
    [ 2, -1, 0, -1, -1, 0, -1, 2],
    [ 2, -1, 0, -1, -1, 0, -1, 2],
    [ 2, -1, 0, 0, 0, 0, -1, 2],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 8, -3, 2, 2, 2, 2, -3, 8]
], dtype=np.int16)


@njit
def moves_generation(board_state, current_color):  # changing to use numba is referenced from AI to have correct numba usage
    opponent_color = -current_color
    legal_move_count = 0

    board_size = board_state.shape[0]
    max_possible_flips = board_size * board_size
    max_possible_moves = board_size * board_size
    

    legal_moves = np.empty((max_possible_moves, 2), np.int8)
    flip_positions = np.empty((max_possible_moves, max_possible_flips, 2), np.int8)
    flip_counts_per_move = np.zeros(max_possible_moves, np.int16)


    for row_index in range(board_size):
        for column_index in range(board_size):
            if board_state[row_index, column_index] != COLOR_NONE:
                continue

            total_flips_for_move = 0

            for direction_index in range(DIRECTIONS_NUMPY.shape[0]):
                row_direction = DIRECTIONS_NUMPY[direction_index, 0]
                column_direction = DIRECTIONS_NUMPY[direction_index, 1]

                current_row = row_index + row_direction
                current_column = column_index + column_direction

                if (current_row < 0)or(current_row >= board_size)or(current_column < 0)or(current_column >= board_size):
                    continue
                if board_state[current_row, current_column] != opponent_color:
                    continue

                direction_start_index = total_flips_for_move
                flips_in_direction = 0

                while (0 <=current_row< board_size) and (0<=current_column< board_size):
                    cell_value = board_state[current_row, current_column]
                    if cell_value == opponent_color:
                        if direction_start_index + flips_in_direction < max_possible_flips:
                            flip_positions[legal_move_count, direction_start_index + flips_in_direction, 0] = current_row
                            flip_positions[legal_move_count, direction_start_index + flips_in_direction, 1] = current_column

                        current_row += row_direction #change
                        current_column += column_direction #change

                        flips_in_direction += 1

                    elif cell_value == current_color:
                        if flips_in_direction > 0:
                            total_flips_for_move += flips_in_direction
                        break
                    else:
                        break

            if total_flips_for_move > 0:
                legal_moves[legal_move_count, 0] = row_index
                legal_moves[legal_move_count, 1] = column_index
                flip_counts_per_move[legal_move_count] = total_flips_for_move
                legal_move_count += 1

    return legal_move_count, legal_moves, flip_positions, flip_counts_per_move


if 'USE_NUMBA' in globals() and USE_NUMBA:
    dummy_board_state = np.zeros((8, 8), dtype=np.int8)
    moves_generation(dummy_board_state, COLOR_BLACK)


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.time_start = 0.0
        self.time_limit = 4.8
        self.time_up = False
        self.best_move = None
        self.current_depth = 0
        self.endgame_empties = 10

    def _search(self, board_state, current_color, search_depth, alpha, beta, previous_player_passed, is_root_node):
        if time.time() - self.time_start > self.time_limit:
            self.time_up = True
            return 0.0

        empty_square_count = board_state.size - np.count_nonzero(board_state != COLOR_NONE)
        if empty_square_count <= self.endgame_empties:
            return self._search_endgame(board_state, current_color, alpha, beta, previous_player_passed, is_root_node)

        legal_move_count, legal_moves_array, flip_positions_array, flip_counts_per_move = moves_generation(board_state, int(current_color))

        if legal_move_count == 0:
            if previous_player_passed:
                return float(self.evaluate_terminal(board_state, current_color))
            return -self._search(board_state, -current_color, search_depth,-beta, -alpha, previous_player_passed=True, is_root_node=False)

        if search_depth == 0:
            return float(self.evaluate(board_state, current_color))

        is_dangerous_move_flags = [
            self._is_dangerous_xsquare(int(legal_moves_array[move_index, 0]), int(legal_moves_array[move_index, 1]), board_state) for move_index in range(legal_move_count)]

        move_indices = list(range(legal_move_count))

        move_indices.sort(
            key=lambda move_index: (
                1 if is_dangerous_move_flags[move_index] else 0,
                int(flip_counts_per_move[move_index])
            )
        )

        if is_root_node and self.best_move is not None:
            best_row, best_column = self.best_move
            for position_in_list, move_index in enumerate(move_indices):
                if int(legal_moves_array[move_index, 0]) == best_row and int(legal_moves_array[move_index, 1]) == best_column:
                    if position_in_list != 0:
                        move_indices.pop(position_in_list)
                        move_indices.insert(0, move_index)
                    break

        best_value = -1000000000.0

        for move_index in move_indices:
            move_row = int(legal_moves_array[move_index, 0])
            move_column = int(legal_moves_array[move_index, 1])

            flip_positions_for_move = flip_positions_array[move_index]

            flip_count_for_move = int(flip_counts_per_move[move_index])

            self._apply_move_inplace(board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move)

            move_value = -self._search(board_state, -current_color, search_depth - 1, -beta, -alpha, previous_player_passed=False, is_root_node=False)

            self._undo_move_inplace(board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move)

            if self.time_up:
                return 0.0

            if move_value > best_value:
                best_value = move_value
                if is_root_node and current_color == self.color:
                    self.best_move = (move_row, move_column)

            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break

        return best_value
    
    def _search_endgame(self, board_state, current_color, alpha, beta, previous_player_passed, is_root_node):
        if ((time.time() - self.time_start) > self.time_limit):
            self.time_up = True
            return 0.0

        legal_move_count, legal_moves_array, flip_positions_array, flip_counts_per_move = moves_generation(board_state, int(current_color))

        if legal_move_count == 0:
            if previous_player_passed:
                return float(self.evaluate_terminal(board_state, current_color))
            return -self._search_endgame(board_state, -current_color,-beta, -alpha,previous_player_passed=True, is_root_node=False)

        is_dangerous_move_flags = [self._is_dangerous_xsquare(int(legal_moves_array[move_index, 0]), int(legal_moves_array[move_index, 1]), board_state) for move_index in range(legal_move_count)]

        move_indices = list(range(legal_move_count))

        move_indices.sort(
            key=lambda move_index: (
                1 if is_dangerous_move_flags[move_index] else 0,
                int(flip_counts_per_move[move_index])
            )
        )

        if is_root_node and self.best_move is not None:
            best_row, best_column = self.best_move
            for position_in_list, move_index in enumerate(move_indices):
                if int(legal_moves_array[move_index, 0]) == best_row and int(legal_moves_array[move_index, 1]) == best_column:
                    if position_in_list != 0:
                        move_indices.pop(position_in_list)
                        move_indices.insert(0, move_index)
                    break

        best_value = -1000000000.0

        for move_index in move_indices:
            move_row = int(legal_moves_array[move_index, 0])
            move_column = int(legal_moves_array[move_index, 1])
            flip_count_for_move = int(flip_counts_per_move[move_index])
            flip_positions_for_move = flip_positions_array[move_index]

            self._apply_move_inplace(board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move)

            move_value = -self._search_endgame(board_state, -current_color, -beta, -alpha, previous_player_passed=False, is_root_node=False)

            self._undo_move_inplace(board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move)

            if self.time_up:
                return 0.0

            if move_value > best_value:
                best_value = move_value
                if is_root_node and current_color == self.color:
                    self.best_move = (move_row, move_column)

            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break

        return best_value

    def _apply_move_inplace(self, board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move):
        board_state[move_row, move_column] = current_color
        for flip_index in range(flip_count_for_move):
            flipped_row = int(flip_positions_for_move[flip_index, 0])
            flipped_column = int(flip_positions_for_move[flip_index, 1])
            board_state[flipped_row, flipped_column] = current_color

    def _undo_move_inplace(self, board_state, current_color, move_row, move_column, flip_positions_for_move, flip_count_for_move):
        board_state[move_row, move_column] = COLOR_NONE
        opponent_color = -current_color
        for flip_index in range(flip_count_for_move):
            flipped_row = int(flip_positions_for_move[flip_index, 0])
            flipped_column = int(flip_positions_for_move[flip_index, 1])
            board_state[flipped_row, flipped_column] = opponent_color

    def _mobility_stats(self, board_state, current_color):
        legal_move_count, _, _, flip_counts_per_move = moves_generation(board_state, int(current_color))

        if legal_move_count == 0:
            return 0, 0.0, 0

        flip_counts_array = np.asarray(flip_counts_per_move[:legal_move_count], dtype=np.int32)

        average_flips = float(flip_counts_array.mean())
        quiet_move_count = int(np.count_nonzero(flip_counts_array <= 2))

        return legal_move_count, average_flips, quiet_move_count

    def evaluate(self, board_state, current_color):
        board_size = self.chessboard_size
        my_disc_mask = (board_state == current_color)
        opponent_disc_mask = (board_state == -current_color)

        my_disc_count = int(np.count_nonzero(my_disc_mask))
        opponent_disc_count = int(np.count_nonzero(opponent_disc_mask))
        total_disc_count = my_disc_count + opponent_disc_count
        empty_square_count = board_size * board_size - total_disc_count

        if total_disc_count == 0:
            disc_score = 0.0
        else:
            disc_score = (opponent_disc_count - my_disc_count) / float(board_size * board_size)

        my_move_count, my_average_flips, my_quiet_move_count = self._mobility_stats(board_state, current_color)
        opponent_move_count, opponent_average_flips, opponent_quiet_move_count = self._mobility_stats(board_state, -current_color)

        if my_move_count + opponent_move_count == 0:
            mobility_score = 0.0
        else:
            mobility_score = (my_move_count - opponent_move_count) / float(my_move_count + opponent_move_count)

        if my_average_flips + opponent_average_flips > 0:
            flip_average_score = (opponent_average_flips - my_average_flips) / 8.0
        else:
            flip_average_score = 0.0

        if my_quiet_move_count + opponent_quiet_move_count > 0:
            quiet_move_score = (opponent_quiet_move_count - my_quiet_move_count) / float(
                my_quiet_move_count + opponent_quiet_move_count
            )
        else:
            quiet_move_score = 0.0

        corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
        my_corner_count = 0
        opponent_corner_count = 0
        for corner_row, corner_column in corners:
            disc_on_corner = board_state[corner_row, corner_column]
            if disc_on_corner == current_color:
                my_corner_count += 1
            elif disc_on_corner == -current_color:
                opponent_corner_count += 1
        if my_corner_count + opponent_corner_count == 0:
            corner_score = 0.0
        else:
            corner_score = (opponent_corner_count - my_corner_count) / float(my_corner_count + opponent_corner_count)

        positional_score = float(np.sum(POSITION_WEIGHT * (opponent_disc_mask.astype(np.int8) - my_disc_mask.astype(np.int8))))
        positional_score /= 100.0

        my_frontier_count = 0
        opponent_frontier_count = 0
        for row_index in range(board_size):
            for column_index in range(board_size):
                cell_value = board_state[row_index, column_index]
                if cell_value == COLOR_NONE:
                    continue
                is_frontier_disc = False
                for row_direction, column_direction in DIRECTIONS:
                    neighbor_row = row_index + row_direction
                    neighbor_column = column_index + column_direction
                    if 0 <= neighbor_row < board_size and 0 <= neighbor_column < board_size and board_state[neighbor_row, neighbor_column] == COLOR_NONE:
                        is_frontier_disc = True
                        break
                if not is_frontier_disc:
                    continue
                if cell_value == current_color:
                    my_frontier_count += 1
                elif cell_value == -current_color:
                    opponent_frontier_count += 1
        if my_frontier_count + opponent_frontier_count == 0:
            frontier_score = 0.0
        else:
            frontier_score = (opponent_frontier_count - my_frontier_count) / float(my_frontier_count + opponent_frontier_count)

        my_stable_count, opponent_stable_count = self.estimate_stable_edges(board_state, current_color)
        if my_stable_count + opponent_stable_count == 0:
            stable_score = 0.0
        else:
            stable_score = (opponent_stable_count - my_stable_count) / float(my_stable_count + opponent_stable_count)

        def compute_disc_spread_area(disc_mask):
            disc_positions = np.argwhere(disc_mask)
            if len(disc_positions) == 0:
                return 0.0
            row_indices = disc_positions[:, 0]
            column_indices = disc_positions[:, 1]
            row_span = int(row_indices.max() - row_indices.min() + 1)
            column_span = int(column_indices.max() - column_indices.min() + 1)
            return float(row_span * column_span)

        my_disc_spread_area = compute_disc_spread_area(my_disc_mask)
        opponent_disc_spread_area = compute_disc_spread_area(opponent_disc_mask)
        if my_disc_spread_area + opponent_disc_spread_area == 0:
            compactness_score = 0.0
        else:
            compactness_score = (opponent_disc_spread_area - my_disc_spread_area) / float(board_size * board_size)

        if empty_square_count >= 44:
            weight_disc_parity = 0.2
            weight_mobility = 7.0
            weight_corners = 5.0
            weight_positional = 5.0
            weight_frontier = 3.0
            weight_stable_discs = 1.0
            weight_compactness = 4.0
            weight_flip_average = 2.0
            weight_quiet_moves = 1.5
        elif empty_square_count >= 20:
            weight_disc_parity = 2.0
            weight_mobility = 6.0
            weight_corners = 8.0
            weight_positional = 5.0
            weight_frontier = 3.0
            weight_stable_discs = 3.0
            weight_compactness = 2.0
            weight_flip_average = 3.0
            weight_quiet_moves = 2.0
        elif empty_square_count >= 10:
            weight_disc_parity = 8.0
            weight_mobility = 3.0
            weight_corners = 9.0
            weight_positional = 3.0
            weight_frontier = 2.0
            weight_stable_discs = 6.0
            weight_compactness = 1.0
            weight_flip_average = 5.0
            weight_quiet_moves = 3.0
        else:
            weight_disc_parity = 25.0
            weight_mobility = 1.0
            weight_corners = 6.0
            weight_positional = 0.5
            weight_frontier = 1.0
            weight_stable_discs = 8.0
            weight_compactness = 0.0
            weight_flip_average = 8.0
            weight_quiet_moves = 4.0

        evaluation_score = (
            weight_disc_parity * disc_score +
            weight_mobility * mobility_score +
            weight_corners * corner_score +
            weight_positional * positional_score +
            weight_frontier * frontier_score +
            weight_stable_discs * stable_score +
            weight_compactness * compactness_score +
            weight_flip_average * flip_average_score +
            weight_quiet_moves * quiet_move_score
        )

        return float(evaluation_score)

    def estimate_stable_edges(self, board_state, current_color):
        board_size = self.chessboard_size
        my_stable_count = 0
        opponent_stable_count = 0
        corner_directions = [(0, 0, 0, 1), (0, 0, 1, 0), (0, board_size - 1, 0, -1), (0, board_size - 1, 1, 0), (board_size - 1, 0, 0, 1), (board_size - 1, 0, -1, 0), (board_size - 1, board_size - 1, 0, -1), (board_size - 1, board_size - 1, -1, 0)]
        for corner_row, corner_column, row_direction, column_direction in corner_directions:
            corner_disc_color = board_state[corner_row, corner_column]
            if corner_disc_color == COLOR_NONE:
                continue
            current_row = corner_row
            current_column = corner_column
            while 0 <= current_row < board_size and 0 <= current_column < board_size:
                if board_state[current_row, current_column] != corner_disc_color:
                    break
                if board_state[current_row, current_column] == current_color:
                    my_stable_count += 1
                elif board_state[current_row, current_column] == -current_color:
                    opponent_stable_count += 1
                current_row += row_direction
                current_column += column_direction
        return my_stable_count, opponent_stable_count

    def _is_dangerous_xsquare(self, row_index, column_index, board_state):
        board_size = self.chessboard_size
        if board_size < 4:
            return False

        if row_index == 1 and column_index == 1:
            return board_state[0, 0] == COLOR_NONE
        if row_index == 1 and column_index == board_size - 2:
            return board_state[0, board_size - 1] == COLOR_NONE
        if row_index == board_size - 2 and column_index == 1:
            return board_state[board_size - 1, 0] == COLOR_NONE
        if row_index == board_size - 2 and column_index == board_size - 2:
            return board_state[board_size - 1, board_size - 1] == COLOR_NONE

        return False

    def evaluate_terminal(self, board_state, current_color):
        my_disc_count = np.count_nonzero(board_state == current_color)
        opponent_disc_count = np.count_nonzero(board_state == -current_color)
        return (opponent_disc_count - my_disc_count) * 10000

    def go(self, chessboard):
        self.candidate_list.clear()
        board_state = chessboard.astype(np.int8, copy=True)

        legal_move_count, legal_moves_array, _, _ = moves_generation(board_state, int(self.color))

        for move_index in range(legal_move_count):
            row_index = int(legal_moves_array[move_index, 0])
            column_index = int(legal_moves_array[move_index, 1])
            self.candidate_list.append((row_index, column_index))

        if legal_move_count == 0:
            return

        self.time_start = time.time()
        if (self.time_out is not None) and (self.time_out > 0):
            self.time_limit = min(self.time_out, 4.8)
        else:
            self.time_limit = 4.8

        self.best_move = (int(legal_moves_array[0, 0]), int(legal_moves_array[0, 1]))

        total_disc_count = int(np.count_nonzero(board_state != COLOR_NONE))
        empty_square_count = self.chessboard_size*self.chessboard_size-total_disc_count

        if empty_square_count > self.endgame_empties:
            if total_disc_count <= 18:
                maximum_search_depth = 6
            elif total_disc_count <= 52:
                maximum_search_depth = 7
            else:
                maximum_search_depth = 16

            search_depth = 1

            while search_depth <= maximum_search_depth:
                self.time_up = False
                self.current_depth = search_depth
                self._search(board_state, self.color, search_depth,-1000000000.0, 1000000000.0, previous_player_passed=False, is_root_node=True)

                if self.time_up:
                    break

                search_depth += 1
        else:
            self.time_up = False
            self.current_depth = 32
            self._search(board_state, self.color, self.current_depth,-1000000000.0, 1000000000.0,previous_player_passed=False, is_root_node=True)

        self.candidate_list.append(self.best_move)