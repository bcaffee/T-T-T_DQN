from TicTacToeEnvironment import TicTacToeEnvironment
from Agent import Agent


def main():
    agent = Agent(TicTacToeEnvironment)
    agent.interactive_game()


if __name__ == "__main__":
    main()
