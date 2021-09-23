class TicTacToeEnvironment:

    def __init__(self):
        self.state = '         '
        self.player = 'X'
        self.winner = None

    def allowed_moves(self):
        states = []
        for i in range(len(self.state)):
            if self.state[i] == ' ':
                states.append(self.state[:i] + self.player + self.state[i + 1:])
        return states

    def handle_agent_turn(self, current_state):
        if self.winner:
            raise (Exception("Game already completed, cannot make another move!"))
        if not self.valid_move(current_state):
            raise (Exception("Cannot make move {} to {} for player {}".format(
                self.state, current_state, self.player)))

        self.state = current_state
        self.winner = self.predict_winner(self.state)

        if self.winner:
            self.player = None
        elif self.player == 'X':
            self.player = 'O'
        else:
            self.player = 'X'

    def handle_human_turn(self, choice):
        self.state = choice
        self.winner = self.predict_winner(self.state)

        if self.winner:
            self.player = None
        elif self.player == 'X':
            self.player = 'O'
        else:
            self.player = 'X'

    def playable(self):
        return (not self.winner) and any(self.allowed_moves())

    @staticmethod
    def predict_winner(current_state):
        winning_orders = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        winner = None
        for winningOrder in winning_orders:
            line_state = current_state[winningOrder[0]] + current_state[winningOrder[1]] + current_state[
                winningOrder[2]]
            if line_state == 'XXX':
                winner = 'X'
            elif line_state == 'OOO':
                winner = 'O'
        return winner

    def valid_move(self, next_state):

        valid = False

        allowed_moves = self.allowed_moves()
        if any(state == next_state for state in allowed_moves):
            valid = True

        return valid

    # For debugging purposes
    def print_board(self):
        s = self.state
        print('     {} | {} | {} '.format(s[0], s[1], s[2]))
        print('    -----------')
        print('     {} | {} | {} '.format(s[3], s[4], s[5]))
        print('    -----------')
        print('     {} | {} | {} '.format(s[6], s[7], s[8]))
        print('\n')
