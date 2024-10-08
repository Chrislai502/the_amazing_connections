

from .endpoints import Endpoint
from .game import Connections, load_json_to_connections, GameOverException

def naive_connections_solver(board) -> list[bool]:
    """
    Solve a Connections game using RSA models.

    :param game: the Connections game to solve
    :return: a list of booleans indicating which levels were solved
    """
    level = 0
    solves = [ False, False, False, False ]

    while level < 4:
        curr_group = board[level]

        guess_word_groups = guess(board, curr_group['members'][0])

        # Check if any of the guessed sets match the target
        for guess in guess_word_groups:

            if sorted(guess.members) == sorted(board['members']) :
                print(f"Level {level} solved! Category: {curr_group['group']}")
                solves[level] = True
                level += 1
                break
        else:
            # If no correct guess, use the top guess and move on (to the next level)
            top_guess = guess_word_groups[0].members
            print(f"All guesses failed!")
            print(f"Top guess: {', '.join(top_guess)}")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    return solves

def script_entrypoint():
    test_connections = load_json_to_connections(filename='test_connections.json')
    word_boards = test_connections.get_words_per_group
    for i in range(10):
        levels_passed = naive_connections_solver(word_boards[i])
    


if __name__ == "__main__":
    script_entrypoint()