from rsallms.solvers.gvc_2 import GVCSolver
from rsallms.game import Connections, load_daily_board, sample_game
import random
import numpy as np
def main():
    # random.seed(41)
    # np.random.seed(41)
    # random.seed(43)
    # np.random.seed(43)
    random.seed(42)
    np.random.seed(42)
    
    game = sample_game()  # Load the game instance
    print(str(game))
    # raise notImplementedError
    solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()

# Category             | Solved? | Words
# ----------------------------------------
# WOODWINDS            |  False  | BASSOON, CLARINET, FLUTE, OBOE
# COVERINGS            |  False  | CAP, COVER, LID, TOP
# SUNGLASSES           |  False  | AVIATOR, CAT EYE, WAYFARER, WRAPAROUND
# SEALS                |  False  | HARBOR, HARP, HOODED, MONK
# 2024-12-11 01:43:48,170 - INFO - GUESSER PROMPT:

# from rsallms.solvers.gvc_2 import GVCSolver
# from rsallms.game import Connections, load_daily_board, sample_game
# import random
# import numpy as np
# def main():
#     # random.seed(41)
#     # np.random.seed(41)
#     # random.seed(42)
#     # np.random.seed(42)
    
#     # Sample 10 games and print out their categories and members
#     for _ in range(10):
#         game = sample_game()  # Load the game instance
#         # print(str(game))
#         for cat in game._og_groups:
#             print(f"{cat.group} -> Words: {cat.members}")
#     raise NotImplementedError
#     solver = GVCSolver()
#     solver.play(game)

# if __name__ == "__main__":
#     main()
