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
    Manages the state of the game session, including the game itself, the solver, and interaction
    with the evaluation database once the game is completed.
    """

    def __init__(self, game: Connections, solver: Solver, max_trials: int = 10, db_path: str = "evaluation_metrics.db"):
        """
        Initialize the GameState.

        :param game: The Connections game instance.
        :param solver: The solver instance that will attempt to solve the game.
        :param max_trials: The maximum number of trials allowed for the solver.
        :param db_path: The file path to the evaluation database.
        """
        self.game = game
        self.solver = solver  # Solver already contains the metrics
        self.max_trials = max_trials
        self.trials = 0
        self.db = EvaluationDatabase(db_filepath=db_path)

    def play_game(self) -> Optional[dict]:
        """
        Plays the game using the solver and records metrics.

        :return: The final game state and metrics after the game completes or None if an error occurs.
        """
        try:
            while not self.game.is_over and self.trials < self.max_trials:
                self.trials += 1
                guess = self.solver.guess(
                    self.game.all_words, self.game.group_size)

                # Simulate solver category matching (you may need to update this)
                # Replace with logic to get the guessed category
                guessed_cat = "placeholder_category"
                cat = self.game.category_guess_check(list(guess))

                if cat is None:
                    self.solver.metrics.increment_failed_guesses()  # Use solver's metrics
                    self.solver.metrics.hallucination_words(
                        list(guess), self.game.all_words)
                else:
                    guessed_cat_idx = self.game._og_groups.index(cat)
                    self.solver.metrics.add_solve(guessed_cat_idx)
                    self.solver.metrics.consine_similarity_category(
                        guessed_cat, cat.group)

            return self.finalize_game()
        except Exception as e:
            print(f"An error occurred during gameplay: {e}")
            return None

    def finalize_game(self) -> dict:
        """
        Finalizes the game, saves metrics to the database, and returns the final game state.

        :return: A dictionary with the game state and metrics.
        """
        metrics = self.solver.metrics  # Use the solver's metrics directly

        game_result = {
            "solves": metrics.solves,
            "failed_guesses": metrics.failed_guesses,
            "solve_order": metrics.solve_order,
            "points": metrics.final_points,
            "solve_rate": metrics.solve_rate,
            "tokens_used": metrics.tokens_used,
            "hallucinated_words": metrics.hallucinated_words
        }

        # Save the metrics in the evaluation database
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

        # Close the database connection
        self.db.close()

        return game_result


# Example usage:
if __name__ == "__main__":
    # Replace with the correct game initialization logic
    game: Connections = load_daily_board()
    solver = NaiveSolver()  # Replace with the actual solver you want to use
    game_state = GameState(game, solver, max_trials=10)

    result = game_state.play_game()
    if result:
        print(f"Final game state: {result}")
    else:
        print("Game failed or encountered an error.")
