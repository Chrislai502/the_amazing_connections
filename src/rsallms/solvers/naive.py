import re

from ..endpoints import Endpoint, get_prompt, EndpointConfig

from .solver import Solver

ENDPOINTS: EndpointConfig = {
    # this is 4 cents per Mil. tok, i.e. free
    "default": lambda metrics: Endpoint("groq", model="llama-3.2-1b-preview", metrics=metrics)
}


class NaiveSolver(Solver):

    @staticmethod
    def _extract_words(response: str) -> list[str]:
        """
        Extract the group of words from the model's response.

        :param response: The model's response
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

        data = {
            "words": ", ".join(word_bank)
        }
        prompt = get_prompt("naive_without_category", **data)

        # TODO: replace the bottom two with a json structured response
        response = ENDPOINTS["default"](self.metrics).respond(prompt)

        return tuple(NaiveSolver._extract_words(response))
