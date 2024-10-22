from ..metrics import Metrics
from ..game import Connections
from ..endpoints import Endpoint, get_prompt, EndpointConfig


ENDPOINTS: EndpointConfig = {
    "default": Endpoint(
        "groq",
        # model="llama-3.2-1b-preview",  # this is 4 cents per Mil. tok, i.e. free
        # model="llama-3.2-3b-preview",
        model="llama-3.2-90b-vision-preview"
    )
}

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
            guessed_cat = "placeholder" # have to figure out how to do this
            cat = game.category_guess_check(list(guess))
            print(f"Guessed: {guess} --> {cat}")

            wrong_guess = cat is None
            if wrong_guess:
                previous_guesses.add(guess)
                self.metrics.hallucination_words(list(guess), game.all_words)
                self.metrics.increment_failed_guesses()
            else:
                guessed_cat_idx = game._og_groups.index(cat)
                # TODO: fix the naming below (this'll probably be super hairy to do)
                self.metrics.add_solve(level=guessed_cat_idx)
                self.metrics.consine_similarity_category(guessed_cat=guessed_cat, correct_cat=cat)
        return game.solved_categories


def extract_words(response: str, word_bank: list[str], group_size: int) -> list[str]:
    prompt_message = f"Given this chat response: {response}, I would like to get the 4 words from the best guess that it has made. Only provide one line of response in this specific format: \"word1 word2 word3 word4\". Nothing else. "
    updated_response = ENDPOINTS["default"].respond(message=prompt_message, temperature=0.1)
                                                    # I would like for you to do the work. Don't provide any code for me to run. Instead just provide me 4 values.")
    print('Connections Guess of 4 words: ', updated_response)
    guess = [
        word for word in word_bank
        if word.upper() in updated_response.upper()
    ]


    if len(guess) != group_size:
        raise ValueError(f"Got improper guess!: {guess}")

    return guess
