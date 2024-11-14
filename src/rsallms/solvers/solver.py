from ..metrics import Metrics
from ..game import Connections
from ..endpoints import Endpoint, EndpointConfig
import time

ENDPOINTS: EndpointConfig = {
    "default": Endpoint(
        "groq",
        # model="llama-3.2-1b-preview",  # this is 4 cents per Mil. tok, i.e. free
        # model="llama-3.2-3b-preview",
        model="llama-3.2-90b-text-preview"
    )
}

class Solver:

    def __init__(self):
        super().__init__()
        if type(self) == Solver:
            raise TypeError("A Solver cannot be instantiated directly!"
                            "Instantiate a subclass instead.")

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None,  history: str = "") -> tuple[tuple[str, ...], str]:
        """
        Guess a set of words that make up a Category in a game of Connections.
        Should instantiate endpoints with the instance attribute `self.metrics`.

        :param word_bank: All of the words remaining in a game of Connections
        :param group_size: The number of words to guess as a group
        :param previous_guesses: The previous failed guesses
        :return: A list of words that make up the guessed category
        """
        raise NotImplementedError

    def play(self, game: Connections, commit_to: str | None = None) -> list[bool]:
        """
        Play a game of Connections.

        :param game: The game to play
        :return: a list of flags indicating which categories were solved
        """
        metrics = Metrics()
        previous_guesses: set[tuple[str, ...]] = set()
        history: str
        history  = ""

        while not game.is_over:
            guess, reasoning = self.guess(
                word_bank=game.all_words,
                group_size=game.group_size,
                previous_guesses=previous_guesses,
                metrics=metrics,
                history=history
            )
            guessed_cat = "placeholder" # have to figure out how to do this
            cat = game.category_guess_check(list(guess))
            print(f"Guessed: {guess} --> {cat}")

            wrong_guess = cat is None
            if wrong_guess:
                previous_guesses.add(guess)
                metrics.hallucination_words(list(guess), game.all_words)
                metrics.increment_failed_guesses()
                if history == "":
                    history += "History: "
                history += "Failed Guess: Word Grouping: " + str(guess) + " Reasoning: ```" + str(reasoning) + "```" + "\n "
            else:
                guessed_cat_idx = game._og_groups.index(cat)
                # TODO: fix the naming below (this'll probably be super hairy to do)
                metrics.add_solve(level=guessed_cat_idx)
                metrics.cosine_similarity_category(guessed_cat=guessed_cat, correct_cat=cat.group)

        if commit_to is not None:
            metrics.commit(to_db=commit_to)
        return game.solved_categories


def extract_words(response: str, word_bank: list[str], group_size: int) -> list[str]:
    """
    Extract guessed words from Agent CoT reasoning
    :return: List of 4 words for Agent's Guess
    """
    prompt_message = f"Given this chat response: {response}, I would like to get the 4 words from the best guess that it has made. Only provide one line of response in this specific format: \"word1 word2 word3 word4\". Nothing else. "
    updated_response = ENDPOINTS["default"].respond(message=prompt_message, temperature=0.1)
                                                    # I would like for you to do the work. Don't provide any code for me to run. Instead just provide me 4 values.")
    print('Connections Guess of 4 words: ', updated_response)
    # guess = [
    #     word for word in word_bank
    #     if word.upper() in updated_response.upper()
    # ]
    guess = updated_response.upper().split()

    # Get a first 4 words that were guessed. If less than 4 words given then fill with empty
    guess = guess[:4] + [''] * (4 - len(guess))
    if len(guess) < group_size:
        guess = guess.append('')
        raise ValueError(f"Got improper guess!: {guess}")

    return guess

def extract_reasoning(response: str, guess: list[str]) -> str:
    """
    Summarize Agent CoT reasoning
    :return:  2-5 word response for the reasoning on why it choose the 4 words for it's guess
    """
    prompt_message = f"Given this chat response: ```{response}```, I would like to get the reasoning that the model used to come up with this guess: ```{guess}```. Please provide a max of 5 word that only correspond to the reasoning for the grouping of this guess: ```{guess}```. Be concise. No more than 5 words. "
    updated_response = ENDPOINTS["default"].respond(message=prompt_message, temperature=0.1)
    print('Connections Reasoning: ', updated_response)
    return updated_response
