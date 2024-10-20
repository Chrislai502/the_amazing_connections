from ..metrics import Metrics
from ..game import Connections


class Solver:

    def __init__(self):
        super().__init__()
        self.metrics = Metrics()
        if type(self) == Solver:
            raise TypeError("A Solver cannot be instantiated directly!"
                            "Instantiate a subclass instead.")

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set()) -> tuple[str, ...]:
        """
        Guess a set of words that make up a Category in a game of Connections.
        Should instantiate endpoints with the instance attribute `self.metrics`.

        :param word_bank: All of the words remaining in a game of Connections
        :param group_size: The number of words to guess as a group
        :param previous_guesses: The previous failed guesses
        :return: A list of words that make up the guessed category
        """
        raise NotImplementedError

    def play(self, game: Connections) -> list[bool]:
        """
        Play a game of Connections.

        :param game: The game to play
        :return: a list of flags indicating which categories were solved
        """
        previous_guesses: set[tuple[str, ...]] = set()
        while not game.is_over:
            guess = self.guess(
                game.all_words,
                game.group_size,
                previous_guesses
            )

            cat = game.guess(list(guess))
            print(f"Guessed: {guess} --> {cat}")

            wrong_guess = cat is None
            if wrong_guess:
                previous_guesses.add(guess)
                self.metrics.increment_failed_guesses()
            else:
                guessed_cat_idx = game._og_groups.index(cat)
                # TODO: fix the naming below (this'll probably be super hairy to do)
                self.metrics.add_solve(level=guessed_cat_idx)
        return game.solved_categories


def extract_words(response: str, word_bank: list[str], group_size: int) -> list[str]:
    guess = [
        word for word in word_bank
        if word in response
    ]

    if len(guess) != group_size:
        raise ValueError(f"Got improper guess!: {guess}")

    return guess
