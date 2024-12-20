
from stateflow import StateFlowGame
from rsallms import Connections
from rsallms import (
    Solver,
    RSASolver,
    NaiveSolver,
    CoTSolver,
    load_games,
    Connections,
    load_daily_board,
)


def main():
    # game = Connections()
    game: Connections = load_daily_board()
    game_flow = StateFlowGame(game)
    game_flow.run()


if __name__ == "__main__":
    main()
