

from .endpoints import Endpoint
from .game import Connections, load_daily_board, GameOverException
from .rsa import PragmaticListener, PragmaticSpeaker, LiteralListener

# Define model configurations
ENDPOINTS = {
    "speaker": Endpoint("http://localhost:11434", model="llama3.2"),
    "pragmatic_listener": Endpoint("http://localhost:11434", model="llama3.2"),
    # this is a bit simpler version compared to the rest
    "literal_listener": Endpoint("http://localhost:11434", model="phi3.5"),
}


def rsa_connections_solver(game: Connections) -> list[bool]:
    """
    Solve a Connections game using RSA models.

    :param game: the Connections game to solve
    :return: a list of booleans indicating which levels were solved
    """
    level = 0
    solves = [False, False, False, False]

    l0 = LiteralListener(game.all_words, ENDPOINTS["literal_listener"])
    s1 = PragmaticSpeaker(game.all_words, ENDPOINTS["speaker"], listener=l0)
    l1 = PragmaticListener(game.all_words, ENDPOINTS["pragmatic_listener"])

    while level < 4:
        curr_group = game.get_groups_by_level(level)[0]

        # Pragmatic Speaker (S1)
        category_utterances = s1.choose_categories(
            curr_group.members, num_samples=1)
        print(f"Generated categories: {category_utterances}")

        # Pragmatic Listener (L1)
        # we could try later on with multiple
        guesses = l1.guess(category_utterances[0], num_samples=1)

        # Check if any of the guessed sets match the target
        for guess in guesses:
            try:
                actual_category = game.guess(guess.members)
            except GameOverException as e:
                raise e

            if actual_category is not None:
                print(f"Level {level} solved! Category: {curr_group.group}")
                solves[level] = True
                level += 1
                break
        else:
            # If no correct guess, use the top guess and move on (to the next level)
            top_guess = guesses[0].members
            print(f"All guesses failed!")
            print(f"Top guess: {', '.join(top_guess)}")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    return solves


def script_entrypoint():
    daily_board = load_daily_board()

    print("Starting RSA algo")
    rsa_score = rsa_connections_solver(daily_board)
    print(f"RSA-inspired solver completed {rsa_score} levels.")


if __name__ == "__main__":
    script_entrypoint()
