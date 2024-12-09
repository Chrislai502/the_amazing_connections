from ..endpoints import Endpoint, generate_prompt, get_prompt
from ..metrics import Metrics
from ..game import Connections
from .solver import Solver, extract_words, extract_reasoning


class CoTSolver(Solver):

    def __init__(self, endpoint_url: str = "groq", model: str = "llama-3.1-70b-versatile"):
        super().__init__()
        self.endpoint = Endpoint(
            endpoint_url,
            model=model
        )

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None, history: str = "") -> tuple[tuple[str, ...], str]:

        num_shots = 0  
        category = None  #if category is None, no category will be given to agent
        prompt = generate_prompt(all_words=word_bank, category=category, num_shots=num_shots, type='cot')
        full_prompt = str(history) + "\n" +  prompt
        print(f"Prompt sent to model:\n{full_prompt}\n")

        system_prompt = get_prompt("system")

        response = self.endpoint.respond(message=full_prompt, system_prompt=system_prompt, metrics=metrics, temperature=0.7)
        

        print(f'Got cot response: "{response}"')

        guess = extract_words(response, word_bank, group_size, metrics=metrics)
        reasoning = extract_reasoning(response, guess, metrics=metrics)

        return tuple(guess), reasoning

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
                    history += "History: \n"
                history += "Failed Guess: " + str(guess) +  " Reasoning: ```" + str(reasoning) + "```" + "\n "
            else:
                guessed_cat_idx = game._og_groups.index(cat)
                metrics.add_solve(level=guessed_cat_idx)
                metrics.cosine_similarity_category(guessed_cat=guessed_cat, correct_cat=cat.group)

        if commit_to is not None:
            metrics.commit(to_db=commit_to)
        return game.solved_categories