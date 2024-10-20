import re

from ..endpoints import Endpoint, get_prompt, EndpointConfig

from .solver import Solver, extract_words

ENDPOINTS: EndpointConfig = {
    "default": lambda metrics: Endpoint(
        "groq",
        # model="llama-3.2-1b-preview",  # this is 4 cents per Mil. tok, i.e. free
        # model="llama-3.2-3b-preview",
        model="llama-3.2-90b-vision-preview",
        metrics=metrics
    )
}


class NaiveSolver(Solver):

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set()) -> tuple[str, ...]:

        data = {
            "words": ", ".join(word_bank)
        }
        prompt = get_prompt("naive_without_category", **data)

        # TODO: replace the bottom two with a json structured response
        response = ENDPOINTS["default"](self.metrics).respond(prompt)

        print(f'Got naive response: "{response}"')
        guess = extract_words(response, word_bank, group_size)

        return tuple(guess)
