from rsallms.solvers.gvc import GVCSolver
from rsallms.game import Connections, load_daily_board, sample_game

def main():
    game = load_daily_board()  # Load the game instance
    solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()
