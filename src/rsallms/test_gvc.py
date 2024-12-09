from rsallms.solvers.gvc_2 import GVCSolver
from rsallms.game import Connections, load_daily_board, sample_game
import random
import numpy as np
def main():
    # random.seed(41)
    # np.random.seed(41)
    random.seed(42)
    np.random.seed(42)
    
    game = sample_game()  # Load the game instance
    print(str(game))
    # raise notImplementedError
    solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()
