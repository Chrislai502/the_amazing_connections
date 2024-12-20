from rsallms.solvers.snap_gvc import SGVCSolver
from rsallms.game import Connections, load_daily_board, sample_game, load_games
import random
import numpy as np
def main():
    # random.seed(41)
    # np.random.seed(41)
    # random.seed(43)
    # np.random.seed(43)
    # random.seed(42)
    # np.random.seed(42)
    
    # game = sample_game()  # Load the game instance
    game =load_games()[-1]
    print(str(game))
    solver = SGVCSolver()
    # solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()
