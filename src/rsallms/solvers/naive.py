
from ..endpoints import Endpoint, generate_prompt, get_prompt, EndpointConfig, EndpointConfig
from ..metrics import Metrics

from .solver import Solver, extract_words, extract_reasoning

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

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None, history: str = "") -> tuple[tuple[str, ...], str]:

        num_shots = 0  
        category = None  #if category is None, no category will be given to agent
        prompt = generate_prompt(all_words=word_bank, category=category, num_shots=num_shots)
        full_prompt = str(history) + "\n" +  prompt
        print(f"Prompt sent to model:\n{full_prompt}\n")
        
        system_prompt = get_prompt("system")

        # TODO: replace the bottom two with a json structured response
        response = ENDPOINTS["default"].respond(message=full_prompt, system_prompt=system_prompt, metrics=metrics, temperature=0.7)
        

        print(f'Got naive response: "{response}"')

        guess = extract_words(response, word_bank, group_size)
        reasoning = extract_reasoning(response, guess)

        return tuple(guess), reasoning

