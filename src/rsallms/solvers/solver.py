from ..game import Connections


class Solver:

    def __init__(self):
        super().__init__()
        if type(self) == Solver:
            raise TypeError("A Solver cannot be instantiated directly!"
                            "Instantiate a subclass instead.")

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set()) -> tuple[str, ...]:
        """
        Guess a set of words that make up a Category in a game of Connections.

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

            corresponding_category = game.guess(list(guess))
            if corresponding_category is None:  # i.e. the guess was wrong
                previous_guesses.add(guess)

        return game.solved_categories
