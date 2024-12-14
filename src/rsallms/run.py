import argparse

from rsallms import (
    Solver,
    BasicSolver,
    NaiveSolver,
    CoTSolver,
    GVCSolver,
    load_games,
    Connections,
)

SOLVERS = {
    'naive': NaiveSolver,
    'cot': CoTSolver,
    'basic': BasicSolver,
    'gvc': GVCSolver,
}


def eval_games(solver: Solver, games: list[Connections], db_name: str):
    for game in games:
        solver.play(game, commit_to=db_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("solver_type", choices=list(SOLVERS.keys()))
    parser.add_argument("model", choices=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gpt-4o-mini"
    ])
    return parser.parse_args()


def main():
    args = parse_args()

    solver = SOLVERS[args.solver_type](
        endpoint_url="groq" if args.model.find("llama") != -1 else "oai",
        model=args.model
    )
    eval_games(
        solver=solver,
        games=load_games()[args.start:args.end],
        db_name="_".join([
            args.solver_type,
            args.model,
            f"{args.start}-{args.end}.db"
        ])
    )


if __name__ == "__main__":
    main()
