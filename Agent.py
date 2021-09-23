import csv
import random
from keras.models import load_model
import numpy as np


# Notes:
# https://www.kaggle.com/dhanushkishore/a-self-learning-tic-tac-toe-program
# https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287


class Agent:

    def __init__(self, game_state, random_aspect=0.1, discount=0.9, agent_tag='X'):

        # Dictionary where the Key is the board state and the value is the reward at that state
        self.value = dict()
        self.NewGame = game_state
        self.randomAspect = random_aspect
        self.discount = discount
        self.agentTag = agent_tag
        self.PLAYER_TAG = 'O'

    # Uses neural net from demoGame()
    def demo_game_stats(self):
        results = []
        for i in range(100):
            results.append(self.demo_game())
        game_stats = {k: results.count(k) / 100 for k in ['X', 'O', '-']}
        print("    percentage results: {}".format(game_stats))

    def learn_game(self, num_episodes):
        for episode in range(num_episodes):
            self.learn_from_episode()

    def learn_from_episode(self):
        game = self.NewGame()

        who_goes_first = random.randint(1, 2)

        # Agent goes first
        if who_goes_first == 1:

            game.player = 'X'

        # Player goes first
        else:
            game.player = 'O'

        # learnSelectMove returns more than one value, and so that is why there is a comma, the "_" means it doesn't
        # matter what the first return value is
        _, move = self.learn_select_move(game)
        while move:
            move = self.learn_from_move(game, move)

    def learn_from_move(self, game, move):
        game.handle_agent_turn(move)
        r = self.reward_or_punish(game)

        next_state_value = 0.0
        selected_next_move = None
        if game.playable():
            best_next_move, selected_next_move = self.learn_select_move(game)
            next_state_value = self.state_value(best_next_move)
        current_state_value = self.state_value(move)
        td_target = r + next_state_value
        self.value[move] = current_state_value + self.discount * (td_target - current_state_value)
        return selected_next_move

    def learn_select_move(self, game):
        allowed_state_values = self.state_values(game.allowed_moves())
        if game.player == self.agentTag:
            best_move = self.get_max_value(allowed_state_values)
        else:
            best_move = self.get_min_value(allowed_state_values)

        selected_move = best_move
        if random.random() < self.randomAspect:
            selected_move = self.random_v(allowed_state_values)

        return best_move, selected_move

    def play_select_move(self, game):
        allowed_state_values = self.state_values(game.allowed_moves())

        # print(allowed_state_values)

        # Accessing the neural network saved in the .h5 file
        model = load_model(
            "....../ML Projects/Reinforcement Learning/TicTacToeReinforcedLearning/CSV_Q-LearningData/TicTacToeModel.h5")

        # Create Numpy array-list of states like a shell shape
        states = np.zeros((1, 9))

        for state in allowed_state_values:
            # axis=0 vertical direction to append, 1 is horizontal
            states = np.append(states, self.convert_state_to_numpy_array(state), axis=0)

        # print(states)

        # Deleting the shell, keeping other rows
        states = np.delete(states, 0, axis=0)

        # print(states)

        # Get numpy list from model prediction
        output_rewards = model.predict(states)

        if game.player == self.agentTag:

            # Find index with highest reward value from the state and get the state, [0] is to get only row indices from the tuple that .where returns
            max_index = np.where(output_rewards == np.amax(output_rewards))[0]

            # Choose randomly from the maximum reward indices, chosen state in numerical terms
            chosen_state = states[np.random.choice(max_index)]

            # print(chosen_state)

            # Hits the first row and different column each time
            graphical_state = self.convert_numpy_array_to_state(chosen_state)

            return graphical_state

        else:

            # Find index with lowest reward value from the state and get the state, [0] is to get only row indices from the tuple that .where returns
            max_index = np.where(output_rewards == np.amin(output_rewards))[0]

            # Choose randomly from the lowest reward indices, chosen state in numerical terms
            chosen_state = states[np.random.choice(max_index)]

            graphical_state = self.convert_numpy_array_to_state(chosen_state)

            return graphical_state

    @staticmethod
    def convert_state_to_numpy_array(state):

        # Passing in tuple (1,9) 1 row, 9 columns
        model_input = np.zeros((1, 9))

        for i in range(9):

            char = state[i]

            if char == 'X':
                model_input[0][i] = 1
            elif char == 'O':
                model_input[0][i] = -1
            else:
                model_input[0][i] = 0

        return model_input

    @staticmethod
    def convert_numpy_array_to_state(np_int_array):

        state = ''

        for i in range(9):

            integer = np_int_array[i]

            # Can't do this because strings aren't mutable in python: state[i] = 'X'

            if integer == 1:
                state += 'X'
            elif integer == -1:
                state += 'O'
            else:
                state += ' '

            # print('&' + state + '&')

        return state

    def demo_game(self, verbose=False):
        game = self.NewGame()
        t = 0

        who_goes_first = random.randint(1, 2)

        # Agent goes first
        if who_goes_first == 1:

            game.player = 'X'

        # Player goes first
        else:
            game.player = 'O'

        while game.playable():
            if verbose:
                print(" \nTurn {}\n".format(t))
                game.print_board()
            move = self.play_select_move(game)
            game.handle_agent_turn(move)
            t += 1
        if verbose:
            print(" \nTurn {}\n".format(t))
            game.print_board()
        if game.winner:
            if verbose:
                print("\n{} is the winner!".format(game.winner))
            return game.winner
        else:
            if verbose:
                print("\nIt's a draw!")
            return '-'

    # Don't use until human interaction in the GUI is set up
    def interactive_game(self):
        game = self.NewGame()
        t = 0

        who_goes_first = random.randint(1, 2)

        # Agent goes first
        if who_goes_first == 1:

            game.player = 'X'

        # Player goes first
        else:
            game.player = 'O'

        while game.playable():
            print(" \nTurn {}\n".format(t))
            game.print_board()

            if game.player == self.agentTag:
                move = self.play_select_move(game)
                game.handle_agent_turn(move)
            else:
                move = self.get_human_move(game)
                game.handle_human_turn(move)

            t += 1

        print(" \nTurn {}\n".format(t))
        game.print_board()

        if game.winner:
            print("\n{} is the winner!".format(game.winner))
            return game.winner
        print("\nIt's a draw!")
        return '-'

    @staticmethod
    def convert_state_to_row(board):

        row = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        for position in range(9):
            if board[position] == 'X':
                row[position] = 1
            elif board[position] == 'O':
                row[position] = -1

        return row

    # For feeding into the neural network
    def save_v_table(self):
        with open('state_values.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['TopLeft', 'TopMid', 'TopRight', 'MidLeft', 'Center', 'MidRight', 'BottomLeft',
                             'BottomMid', 'BottomRight', 'RewardValue'])

            all_states = list(self.value.keys())
            all_states.sort()

            for state in all_states:
                numeric_board_state = self.convert_state_to_row(state)
                reward_value = self.value[state]
                numeric_board_state.append(reward_value)
                writer.writerow(numeric_board_state)

    # For having cleaner floats
    def round_v(self):
        for k in self.value.keys():
            self.value[k] = round(self.value[k], 1)

    def state_value(self, game_state):
        return self.value.get(game_state, 0.0)

    def state_values(self, game_states):
        return dict((state, self.state_value(state)) for state in game_states)

    @staticmethod
    def get_max_value(state_values):
        max_v = max(state_values.values())
        chosen_state = random.choice([state for state, v in state_values.items() if v == max_v])
        return chosen_state

    @staticmethod
    def get_min_value(state_values):
        min_v = min(state_values.values())
        chosen_state = random.choice([state for state, v in state_values.items() if v == min_v])
        return chosen_state

    @staticmethod
    def random_v(state_values):
        return random.choice(list(state_values.keys()))

    def reward_or_punish(self, game):

        # self.value_player is agent
        if game.winner == self.agentTag:
            return 1.0

        elif game.winner:
            return -1.0

        # Stalemate
        else:
            return 0.0

    @staticmethod
    def get_human_move(game):
        allowed_moves = [i + 1 for i in range(9) if game.state[i] == ' ']
        human_move = None
        while not human_move:

            user_input = ""

            while user_input is not int:

                try:
                    # User input gets taken in as whatever type was entered, so cast is needed
                    user_input = int(input('Please choose move for {}, from {}: '.format(game.player, allowed_moves)))
                    break

                except ValueError:

                    print("Please enter a valid number.")

                except NameError:

                    print("Please enter a valid number.")

                except SyntaxError:

                    print("Please enter a valid number.")

            idx = int(user_input)

            if any([i == idx for i in allowed_moves]):
                # Means everything before or after depending (before - inclusive, after - exclusive)
                human_move = game.state[:idx - 1] + game.player + game.state[idx:]

        return human_move
