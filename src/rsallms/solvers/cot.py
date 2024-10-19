import re

from typing import Literal

from ..endpoints import Endpoint, CannedResponder, get_prompt

from .solver import Solver


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
ENDPOINTS = {
    # Adjust the model and endpoint URL as needed
    "cot_model": CannedResponder(canned_response)
}


class CoTSolver(Solver):
    """
    Solve a Connections game using Chain-of-Thought (CoT) prompting,
    supporting both zero-shot and one-shot modes.
    """

    @staticmethod
    def _get_prompt(words: list[str], shot_type: Literal["zero-shot", "one-shot"] = "zero-shot") -> str:
        """
        Generate a Chain-of-Thought (CoT) prompt for the given words, using Mustache templates.

        :param words: List of words for the CoT reasoning prompt
        :param include_category: Boolean, whether the agent knows the category it is trying to group
        :param shot_type: Specifies whether to use zero-shot or one-shot prompting ("zero-shot" or "one-shot")
        :param category: The category to be passed into the prompt if include_category is True
        :return: The generated CoT prompt
        """
        # Prepare the data to pass into the Mustache template
        words_str = ', '.join(words)

        # Use the template for not including category
        data = {
            "words": words_str
        }
        prompt = get_prompt("naive_without_category", **data)

        # Load one-shot example from Mustache file
        one_shot_example = get_prompt("one_shot_without_category")

        # If shot_type is one-shot, prepend the one-shot example to the actual prompt
        if shot_type == "one-shot":
            prompt = one_shot_example + prompt

        return prompt

    @staticmethod
    def _extract_words(response: str) -> list[str]:
        """
        Extract the group of words from the CoT model's response.

        :param response: The CoT model's response
        :return: A list of four guessed words
        """
        # Regular expression to find words between the phrase "belong to" or "are" and "category" or the end of the sentence.
        # This assumes the model outputs something like: "Apple, banana, orange, and grape belong to the category..."
        pattern = r"(\b\w+\b(?:,? and)?){4}"

        # Search for the four words in the response
        match = re.search(pattern, response)

        if match:
            # Extract the words and split them into a list, handling commas and conjunctions
            words = re.findall(r'\b\w+\b', match.group(0))
            return words
        else:
            # If no match is found, return an empty list or handle it appropriately
            return []

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set()) -> tuple[str, ...]:
        cot_prompt = CoTSolver._get_prompt(word_bank, "one-shot")

        reasoning = ENDPOINTS['cot_model'].respond(cot_prompt)

        print(f"Generated category reasoning: {reasoning}")

        # CoT Guessing: Extract guessed words from CoT response
        guess = CoTSolver._extract_words(reasoning)

        return tuple(guess)
