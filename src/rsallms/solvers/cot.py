import re

from typing import Literal

from ..endpoints import EndpointConfig, Endpoint, chain_prompts
from ..metrics import Metrics

from .solver import Solver, extract_words


def canned_response(msg: str, sys_msg: str | None) -> str:
    """
    Simulated response from the CoT model for the provided prompt. This is useful for testing.
    """

    print(f"Prompt sent to model:\n{msg}\n")

    if "category" in msg:
        return "Apple, banana, orange, and grape belong to the category of 'fruits.' These are all edible, natural products."
    else:
        return "Apple, banana, orange, and grape all share a common characteristic: they are types of fruit."


# Define model configuration for CoT prompting
ENDPOINTS: EndpointConfig = {
    # "default": lambda metrics: CannedResponder(canned_response)
    "default": Endpoint(
        "groq",
        # model="llama-3.2-3b-preview",
        model="llama-3.2-90b-vision-preview",
    )
}


class CoTSolver(Solver):
    """
    Solve a Connections game using Chain-of-Thought (CoT) prompting,
    supporting both zero-shot and one-shot modes.
    """

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None) -> tuple[tuple[str, ...], str]:
        cot_prompt = chain_prompts(
            [
                "one_shot_without_category",
                "naive_without_category"
            ],
            words=', '.join(word_bank)
        )

        reasoning = ENDPOINTS['default'].respond(cot_prompt, metrics=metrics)

        print(f"Generated category reasoning: {reasoning}")

        # CoT Guessing: Extract guessed words from CoT response
        guess = extract_words(reasoning, word_bank, group_size)

        return tuple(guess), reasoning
