from rsallms import (
    Solver,
    RSASolver,
    NaiveSolver,
    CoTSolver,
    load_games,
    Connections,
    load_daily_board,
)
import sqlite3 as dbms
EVAL_DB = "evals.db"
solver: Solver = (
    NaiveSolver(),
    CoTSolver(),
    RSASolver()
)[0]

games: list[Connections] = load_games()
for i in range(32, 100):
    game = games[i]
    solver.play(game, commit_to=EVAL_DB)
