from rsallms.solvers.gvc import GVCSolver
from rsallms.game import Connections, load_daily_board, sample_game, load_games

def main():
    games = load_games()
    game = games[20]  # Load the game instance
    solver = GVCSolver()
    solver.play(game)

if __name__ == "__main__":
    main()
