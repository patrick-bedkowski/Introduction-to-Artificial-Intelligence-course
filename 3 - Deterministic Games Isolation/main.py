import pandas as pd
import numpy as np

import random
import copy

import sys
import os
import time
import tabulate as tb
from typing import Tuple, List

class NoLegalMoves(Exception):
    """
    Custom exception meant to indicate whether the player does not have any moves left.
    """
    def __init__(self, _player):
        self.player = _player

"""
    # ----------------------- #
    #       Game Agents       #
    # ----------------------- #
"""

class MinimaxPlayer:
    """
    Implementation of an agent playing isolation game with minimax algorithm.
    """
    def __init__(self,
                 starting_position,
                 search_depth=4):
        self.search_depth = search_depth
        self.position = starting_position
        self._used_positions = [starting_position]

    def get_n_moves(self) -> int:
        """
        Returns number of moves of a player.
        """
        return len(self._used_positions)-1

    def get_best_move(self, game, current_player) -> Tuple[int, int]:
        """
        Search for the best move from the available legal moves.
        :param game: An instance of Isolation object that encodes current state of the game
        :param current_player: Player type name
        :return: Tuple with best move coordinates
        """
        score, move = self.minimax(game, self.search_depth, current_player)

        if move is None:
            raise NoLegalMoves(self)

        return move

    def minimax(self, future_game, depth, current_player) -> Tuple[any, any]:
        """
        Implementation of an minimax algorithm for agent playing Isolation game.
        :param future_game: Next instance of a isolation game
        :param depth: Available depth of a decision tree.
        :param current_player:Player type name.
        :return: Tuple with best score and best move
        """

        # get legal moves of a player
        legal_moves = future_game.get_legal_moves(current_player)

        if len(legal_moves) > 0:  # if player has any future moves
            if depth > 0:
                if current_player == 'max':  # :  # MAXIMIZING player

                    best_score, best_move = None, None
                    for move in legal_moves:
                        new_score, _ = self.minimax(future_game.forecast_move(move, current_player), depth - 1, 'min')
                        if best_score is None or new_score > best_score:
                            best_score, best_move = new_score, move

                else:  # MINIMIZING player
                    best_score, best_move = None, None
                    for move in legal_moves:
                        new_score, _ = self.minimax(future_game.forecast_move(move, current_player), depth - 1, 'max')
                        if best_score is None or new_score < best_score:
                            best_score, best_move = new_score, move
            else:
                best_score, best_move = future_game.score(), None
        else:
            best_score, best_move = future_game.score(), None

        return best_score, best_move


class RandomPlayer:
    """
    Implementation of an agent playing isolation game in a random manner.
    """
    def __init__(self,
                 starting_position,
                 search_depth=3):
        self.search_depth = search_depth
        self.position = starting_position
        self._used_positions = [starting_position]

    def get_n_moves(self):
        """
        Returns number of moves of a player.
        """
        return len(self._used_positions)-1

    def get_best_move(self, game, current_player):
        """
        Search for the best move from the available legal moves.
        :param current_player: Name type of current player.
        :param game: Game instance
        :return:
        """
        legal_moves = game.get_legal_moves(current_player)

        if len(legal_moves) == 0:
            raise NoLegalMoves(self)

        return random.choice(legal_moves)


"""
    # ------------------------- #
    #       Isolation Game      #
    # ------------------------- #
"""


class IsolationGame:
    def __init__(self, size, _player1, _player2, _depth, start_player: str = 'player1', _plot=False):
        """
        Implementation of an isolation game.
        :param size: board size.
        :param _player1: Type of player object defined in this file.
        :param _player2: Type of player object defined in this file.
        :param _depth: Maximum depth used in minimax algorithm.
        :param start_player: Which player should start the game.
        :param _plot: If true, plots are visible on the output console.
        """
        self.player1 = _player1  # O
        self.player2 = _player2  # X
        self.start_player(start_player)  # initialize default starting player

        """
        Game parameters
        """
        self._size = size
        self.plot = _plot
        self.player1.search_depth = _depth
        self.player2.search_depth = _depth
        self._scene = self.create_scene()

    def create_scene(self) -> np.array:
        """
        Creates new scene of shape (size x size)
        """
        scene = np.full(shape=(self._size, self._size), dtype=str, fill_value='-')

        x = self.player1.position[0]
        y = self.player1.position[1]
        scene[x][y] = 'O'

        x = self.player2.position[0]
        y = self.player2.position[1]
        scene[x][y] = 'X'
        return scene

    def start_player(self, player: str) -> None:
        """
        Manages which players should start the game.
        :param player: player name
        """
        if player == 'player1':
            self.active_player_p = self.player1  # maximizing player
            self.inactive_player_p = self.player2  # minimizing player
            self.active_player = 'max'
        else:
            self.active_player_p = self.player2  # maximizing player
            self.inactive_player_p = self.player1  # minimizing player
            self.active_player = 'min'

    def get_opponent(self, player):
        """
        Returns an opponent of a current player.
        :param player: Player type
        :return: Reference to player object
        """
        if player != 'max':
            return 'min'
        elif player != 'min':
            return 'max'
        else:
            print("No opponent was returned")
            return None

    def is_loser(self, player) -> bool:
        """
        Test whether the specified player has lost the game.
        :param player: player name
        :return: Boolean value of a condition
        """
        return len(self.get_legal_moves(player)) == 0

    def is_winner(self, player) -> bool:
        """
        Test whether the specified player has won the game.
        :param player: player name
        :return: Boolean value of a condition
        """
        return len(self.get_legal_moves(player)) != 0 and len(self.get_legal_moves(self.get_opponent(player))) == 0

    def move_player(self, move: Tuple[any]) -> None:
        """
        Applies the move of a player on the scene.
        :param move: Tuple with move coordinates.
        """
        # mark last used position on the scene
        if len(self.active_player_p._used_positions) > 0:
            last_position = self.active_player_p._used_positions[-1]
            self._scene[last_position[0]][last_position[1]] = '|'

        if self.active_player_p == self.player1:
            self._scene[move[0]][move[1]] = 'O'
        elif self.active_player_p == self.player2:
            self._scene[move[0]][move[1]] = 'X'

        self.active_player_p.position = move
        self.active_player_p._used_positions.append(move)

    def plot_scene(self) -> None:
        """
        Outputs the scene into the terminal window.
        This function uses os module with commands that are specific for the certain os.
        :return: None
        """
        if sys.platform == 'win32':
            os.system('cls')  # clear the terminal screen
        else:
            os.system('clear')

        # display scene
        print(tb.tabulate(self._scene, tablefmt="fancy_grid"))

    def get_legal_moves(self, player) -> List[any]:
        """
        Returns a list with available moves for a player
        :param player: player type name
        :return: List with available moves
        """
        if player == 'max':
            x = self.player1.position[0]
            y = self.player1.position[1]
        else:
            x = self.player2.position[0]
            y = self.player2.position[1]

        legal_moves = []
        for j in range(y - 1, y + 2):
            for i in range(x - 1, x + 2):
                if 0 <= i < len(self._scene) and 0 <= j < len(self._scene[0]) \
                        and self._scene[i][j] == '-' and (x, y) != (i, j):
                    legal_moves.append((i, j))
        return legal_moves

    def change_player(self, current_player):
        """
        Change game attributes describing which player is active and inactive.
        :param current_player: Name type of current player
        """
        if current_player == 'max':
            self.active_player_p = self.player2
            self.inactive_player_p = self.player1
            self.active_player = 'min'
        elif current_player == 'min':
            self.active_player_p = self.player1
            self.inactive_player_p = self.player2
            self.active_player = 'max'
        else:
            print("PLAYER HAS NOT BEEN CHANGED!")

    def forecast_move(self, move, current_player):
        """
        Return a copy of the game with applied move.
        :return: Board with applied move
        """
        new_board = copy.deepcopy(self)  # make copy of a board
        new_board.move_player(move)  # apply new move

        new_board.change_player(current_player)

        return new_board

    def score(self) -> float:
        """
        Scores the game.
        :return: Float score of a game.
        """

        if game.active_player == 'max':  # Maximizer
            if game.is_loser('max'):
                return float("-inf")
            if game.is_winner('max'):
                return float("inf")
        else:
            if game.is_loser('min'):
                return float("inf")
            if game.is_winner('min'):
                return float("-inf")

        player1_moves = len(self.get_legal_moves('max'))
        player2_moves = len(self.get_legal_moves('min'))

        return float(3*player1_moves - player2_moves)

    def start_game(self, exec_delay: bool = False):
        """
        Main method initializing the game.
        :param exec_delay: Sets the delay between moves.
        :return:
        """
        try:
            # plot beginning scene
            if self.plot:
                self.plot_scene()

            # start game
            while True:
                game_copy = copy.deepcopy(self)

                curr_move = self.active_player_p.get_best_move(game_copy, self.active_player)

                self.move_player(curr_move)

                if exec_delay:
                    time.sleep(1)

                if self.plot:
                    self.plot_scene()

                self.change_player(self.active_player)

        except NoLegalMoves as nol:
            if self.player1 == nol.player:
                # player 1 has lost
                return 'player2'  # return winner
            if self.player2 == nol.player:
                # player 2 has lost
                return 'player1'  # return winner


def generate_starting_point(BOARD_SIZE: int) -> Tuple[int, int]:
    """
    Generates a random starting point from the board
    :param BOARD_SIZE: Size of a board
    :return:
    """
    return random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)


def custom_print(message, TEST_PLAYERS):
    if not TEST_PLAYERS:
        print(message)

def initialize_players(mode: str, TEST_PLAYERS):
    # Create MinimaxPlayer and RandomPlayer
    p1_position = generate_starting_point(BOARD_SIZE)
    player1 = MinimaxPlayer(p1_position)

    p2_position = generate_starting_point(BOARD_SIZE)
    while p2_position == p1_position:  # while position is not different than of player1
        p2_position = generate_starting_point(BOARD_SIZE)

    if mode == 'minimax':
        player2 = MinimaxPlayer(p2_position)
        custom_print("\nGame initialized with two MiniMax players", TEST_PLAYERS)
    elif mode == 'mixed':
        player2 = RandomPlayer(p2_position)
        custom_print("\nGame initialized with a MiniMax player and a Random player", TEST_PLAYERS)
    else:
        custom_print("Unspecified mode. Program will be run with default option...", TEST_PLAYERS)
        player2 = RandomPlayer(p2_position)

    if mode == 'minimax':
        custom_print("O - MiniMax player - Maximizer", TEST_PLAYERS)
        custom_print("X - MiniMax player - Minimizer", TEST_PLAYERS)
    else:
        custom_print("O - MiniMax player", TEST_PLAYERS)
        custom_print("X - Random player", TEST_PLAYERS)

    return player1, player2

"""
    # ----------------------- #
    #      Main Program       #
    # ----------------------- #
"""


if __name__ == '__main__':

    BOARD_SIZE = 5
    DEPTH = 3
    TEST_PLAYERS = False  # If true, enables additional features used for testing large amount of games
    chosen_mode = None

    if len(sys.argv) == 2:
        chosen_mode = sys.argv[1]
        player1, player2 = initialize_players(chosen_mode, TEST_PLAYERS)
    else:
        print("No mode was selected. Program will be run with default option...")
        player1, player2 = initialize_players('mixed', TEST_PLAYERS)

    if not TEST_PLAYERS:

        STARTING_PLAYER = 'player1'

        game = IsolationGame(BOARD_SIZE, player1, player2, DEPTH, STARTING_PLAYER, True)

        time.sleep(3)  # for readability

        time_start = time.time()
        player_won = game.start_game(exec_delay=True)
        exec_time = time.time() - time_start

        n_of_moves = game.player1.get_n_moves() + game.player2.get_n_moves()

        print(f'\n{player_won.capitalize()} ({"RandomPlayer" if player_won == "player2" and chosen_mode == "mixed" else "MinimaxPlayer"}) has won the game')
        print('\nTotal number of moves: ', n_of_moves)
        print('Execution time with subtracted delay: {:.2f} [s]'.format(exec_time-1*n_of_moves))  # exec_time - delay*number_of_moves

    # ============================= #
    #       Testing playground      #
    # ----------------------------- #
    # It is used to run high        #
    # number of games, with results #
    # being saved in a form of      #
    # .csv file                     #
    # ============================= #

    else:
        df_columns = ['who_won', 'starting_player', 'depth', 'exec_time', 'position_p1', 'position_p2']

        depth_params = [2, 3, 4]
        starting_player = ['player1', 'player2']

        data_scores = {}

        for _depth in depth_params:
            scores_df = pd.DataFrame(columns=df_columns)
            for s_player in starting_player:
                for _ in range(0, 100):  # run tests multiple times
                    player1, player2 = initialize_players('minimax', TEST_PLAYERS)

                    game = IsolationGame(BOARD_SIZE, player1, player2, _depth, s_player, False)

                    # get players position
                    data_scores['position_p1'] = str(game.player1.position)
                    data_scores['position_p2'] = str(game.player2.position)

                    time_start = time.time()
                    player_won = game.start_game()
                    exec_time = time.time() - time_start

                    data_scores['who_won'] = player_won
                    data_scores['starting_player'] = s_player
                    data_scores['depth'] = _depth
                    data_scores['exec_time'] = exec_time

                    print(f'{player_won.capitalize()} has won the game')

                    scores_df = scores_df.append(data_scores, ignore_index=True)
            scores_df.to_excel(f'scores_MINIMAX_{_depth}.xls', index=False)
