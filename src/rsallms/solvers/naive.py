import re

from ..endpoints import Endpoint, get_prompt, EndpointConfig, prepare_examples, EndpointConfig
from ..metrics import Metrics

from .solver import Solver, extract_words

ENDPOINTS: EndpointConfig = {
    "default": Endpoint(
        "groq",
        # model="llama-3.2-1b-preview",  # this is 4 cents per Mil. tok, i.e. free
        # model="llama-3.2-3b-preview",
        # model="llama-3.2-90b-vision-preview",
        model="llama-3.1-70b-versatile"
    )
}


class NaiveSolver(Solver):

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None) -> tuple[str, ...]:

        num_shots = 0  
        category = None  #if category is None, no category will be given to agent
        prompt = generate_prompt(all_words=word_bank, category=category, num_shots=num_shots)

        
        system_prompt = get_prompt("system")

        # TODO: replace the bottom two with a json structured response
        response = ENDPOINTS["default"].respond(message=prompt, system_prompt=system_prompt, metrics=metrics)


        print(f'Got naive response: "{response}"')

        guess = extract_words(response, word_bank, group_size)

        return tuple(guess)

