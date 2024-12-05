from rsallms.solvers.gvc_2 import GVCSolver
from rsallms.game import Connections, load_daily_board, sample_game

def main():
    game = sample_game()  # Load the game instance
    solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()
