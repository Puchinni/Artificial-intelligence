from copy import deepcopy
from connect4 import Connect4

class MinMaxAgentHeur:
    def __init__(self, my_token):
        self.my_token = my_token
        self.opponent_token = 'o' if my_token == 'x' else 'x'
        self.max_depth = 4

    def decide(self, connect4: Connect4):
        return self.minimax(connect4, self.max_depth, True)[1]

    def heurystyka(self, connect4: Connect4, maximizing_player):
        probability = connect4.amount_of_threats(maximizing_player)
        return probability, None

    def minimax(self, connect4: Connect4, depth, maximizing_player):
        if depth == 0:
            return self.heurystyka(connect4, maximizing_player)
        
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
                    new_value = self.minimax(new_state, depth - 1, False)[0]
                    if new_value > value:
                        value = new_value
                        column = move
            return value, column
        else:
            value = float('inf')
            column = None
            for move in connect4.possible_drops():
                    new_state = deepcopy(connect4)
                    new_state.drop_token(move)
                    new_value = self.minimax(new_state, depth - 1, True)[0]
                    if new_value < value:
                        value = new_value
                        column = move
            return value, column
        
    