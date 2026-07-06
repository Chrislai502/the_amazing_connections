from ..endpoints import Endpoint, generate_prompt
from ..metrics import Metrics

from .solver import Solver, extract_words, extract_guessed_category


class BasicSolver(Solver):

    def __init__(self, endpoint_url: str = "groq", model: str = "llama-3.3-70b-versatile"):
        super().__init__()
        self.endpoint = Endpoint(
            endpoint_url,
            model=model
        )

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None, history: str = "") -> tuple[tuple[str, ...], str]:

        num_shots = 0  
        category = None  #if category is None, no category will be given to agent
        prompt = generate_prompt(all_words=word_bank, category=category, num_shots=num_shots, type='basic')
        full_prompt = str(history) + "\n" +  prompt

        response = self.endpoint.respond(message=full_prompt, system_prompt=None, metrics=metrics, temperature=0.7)

        guess = extract_words(response, word_bank, group_size, metrics=metrics)
        guessed_category = extract_guessed_category(response, guess, metrics=metrics)

        return tuple(guess), guessed_category

