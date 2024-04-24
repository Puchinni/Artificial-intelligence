from copy import deepcopy
from connect4 import Connect4

class AlphaBetaAgent:
    def __init__(self, my_token):
        self.my_token = my_token
        self.opponent_token = 'o' if my_token == 'x' else 'x'
        self.max_depth = 5

    def decide(self, connect4: Connect4):
        return self.alphabeta(connect4, self.max_depth, float('-inf'), float('inf'), True)[1]

    def alphabeta(self, connect4: Connect4, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return 0, None
        
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1, None
            elif connect4.wins == self.opponent_token:
                return -1, None
            else:
                return 0, None

        if maximizing_player:
            value = float('-inf')
            column = None
            for move in connect4.possible_drops():
                    new_state = deepcopy(connect4)
                    new_state.drop_token(move)
                    new_value = self.alphabeta(new_state, depth - 1, alpha, beta, False)[0]
                    if new_value > value:
                        value = new_value
                        column = move
                    alpha = max(alpha, value)
                    if value >= beta:
                        break
            return value, column
        else:
            value = float('inf')
            column = None
            for move in connect4.possible_drops():
                    new_state = deepcopy(connect4)
                    new_state.drop_token(move)
                    new_value = self.alphabeta(new_state, depth - 1, alpha, beta, True)[0]
                    if new_value < value:
                        value = new_value
                        column = move
                    beta = min(beta, value)
                    if alpha >= value:
                        break
            return value, column