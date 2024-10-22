from typing import Optional

from rsallms import (
    Solver,
    RSASolver,
    NaiveSolver,
    CoTSolver,

    Connections,
    EvaluationDatabase,
    load_daily_board,
)


class GameState:
    """
    Manages the state of the game session, including multiple gameplay trials, each with a new solver instance,
    and interaction with the evaluation database after each gameplay.
    """

    def __init__(self, game: Connections, solver_class: type(Solver), max_trials: int = 10, db_path: str = "evaluation_metrics.db"):
        """
        Initialize the GameState.

        :param game: The Connections game instance.
        :param solver_class: The solver class that will be instantiated for each trial.
        :param max_trials: The maximum number of gameplay trials.
        :param db_path: The file path to the evaluation database.
        """
        self.game = game
        # The solver class (e.g., NaiveSolver)
        self.solver_class = solver_class
        self.max_trials = max_trials
        self.trials = 0
        self.db = EvaluationDatabase(db_filepath=db_path)

    def play_games(self) -> Optional[list[dict]]:
        """
        Plays multiple games, each with a new solver, and records metrics after each gameplay trial.

        :return: A list of game state and metrics after each game trial, or None if an error occurs.
        """
        all_game_results = []  # To store results of all trials
        try:
            while self.trials < self.max_trials:
                self.trials += 1

                # Reset the game to its original state
                self.game.reset()

                # Create a new solver instance for this trial
                solver = self.solver_class()

                # Play the game with the current solver
                self.play_single_game(solver)

                # Finalize the game and save the metrics
                game_result = self.finalize_game(solver)
                all_game_results.append(game_result)

            return all_game_results
        except Exception as e:
            print(f"An error occurred during gameplay: {e}")
            return None

    def play_single_game(self, solver: Solver):
        """
        Plays a single game with the provided solver instance.

        :param solver: The solver instance used for this game.
        """
        while not self.game.is_over:
            guess = solver.guess(self.game.all_words, self.game.group_size)

            # Simulate solver category matching (you may need to update this)
            guessed_cat = "placeholder_category"
            cat = self.game.category_guess_check(list(guess))

            if cat is None:
                solver.metrics.increment_failed_guesses()  # Use solver's metrics
                solver.metrics.hallucination_words(
                    list(guess), self.game.all_words)
            else:
                guessed_cat_idx = self.game._og_groups.index(cat)
                solver.metrics.add_solve(guessed_cat_idx)
                solver.metrics.consine_similarity_category(
                    guessed_cat, cat.group)

    def finalize_game(self, solver: Solver) -> dict:
        """
        Finalizes a single game trial, saves metrics to the database, and returns the final game state.

        :param solver: The solver instance used for this game.
        :return: A dictionary with the game state and metrics for this gameplay trial.
        """
        metrics = solver.metrics  # Use the solver's metrics directly

        game_result = {
            "solves": metrics.solves,
            "failed_guesses": metrics.failed_guesses,
            "solve_order": metrics.solve_order,
            "points": metrics.final_points,
            "solve_rate": metrics.solve_rate,
            "tokens_used": metrics.tokens_used,
            "hallucinated_words": metrics.hallucinated_words
        }

        # Save the metrics in the evaluation database after each trial
        self.db.add_evaluation_entry(
            hallucination_rate=metrics.hallucinated_words,
            num_failed_guesses=metrics.failed_guesses,
            solve_rate=metrics.solve_rate,
            solve_order=metrics.solve_order,
            num_tokens_generated=sum(t['completion_tokens']
                                     for t in metrics.tokens_used.values()),
            num_tokens_ingested=sum(t['prompt_tokens']
                                    for t in metrics.tokens_used.values())
        )

        return game_result


# Example usage:
if __name__ == "__main__":
    # Replace with the correct game initialization logic
    game: Connections = load_daily_board()

    # Initialize the GameState with NaiveSolver (you can replace with RSASolver or CoTSolver)
    game_state = GameState(game, NaiveSolver, max_trials=10)

    # Play the games
    results = game_state.play_games()
    if results:
        for idx, result in enumerate(results, start=1):
            print(f"Game trial {idx} results: {result}")
    else:
        print("Game failed or encountered an error.")
