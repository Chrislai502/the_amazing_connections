
from heapq import heappush, heappop
from collections.abc import Generator

from ..game import Category
from ..endpoints import get_prompt, Endpoint, EndpointConfig
from ..metrics import Metrics

from .solver import Solver

# Define model configurations
ENDPOINTS: EndpointConfig = {
    "speaker": Endpoint("http://localhost:11434", model="llama3.2"),
    "pragmatic_listener": Endpoint("http://localhost:11434", model="llama3.2"),
    # this is a bit simpler version compared to the rest
    "literal_listener": Endpoint("http://localhost:11434", model="phi3.5"),
}


class Listener:
    def __init__(self, all_words: list[str], endpoint: Endpoint):
        super().__init__()
        self.endpoint = endpoint
        self.all_words = all_words

    def guess(self, category: str, num_samples: int = 1, metrics: Metrics | None = None) -> list[Category]:
        """
        Produce a collection of guesses for the set of target words that the speaker intended to communicate given a category.

        :param category: the category given by the speaker
        :param num_samples: [optional] the number of samples to generate, defaults to 1
        :return: a list of Category objects, each containing a set of target words that the speaker intended
        """
        raise NotImplementedError

    def evaluate_category(self, category: str, target_words: list[str], metrics: Metrics | None = None) -> int:
        """
        Evaluate the quality of a given category for describing a set of target
        words by computing the number of target words that are also guessed by
        a literal listener given that category.

        In essence, this is P(L interprets target_words out of all_words | category)

        :param category: the category to evaluate
        :return: the number of target words that are also guessed by a literal listener given this category
        """
        guess = set(self.guess(category, metrics=metrics))
        return len(guess.intersection(set(target_words)))


class LiteralListener(Listener):
    def guess(self, category: str, num_samples: int = 1, metrics: Metrics | None = None) -> list[Category]:
        if num_samples != 1:
            raise ValueError("num_samples must be 1 for literal listeners")

        response = self.endpoint.respond(
            message=get_prompt("L0", category=category,
                               all_words=', '.join(self.all_words)),
            system_prompt="You are a literal interpreter of language. Don't overthink or look for hidden meanings.",
            metrics=metrics
        )
        return [
            Category(level=-1, group=category,
                     members=response.strip().split(", "))
        ]


class PragmaticListener(Listener):
    def guess(self, category: str, num_samples: int = 1, metrics: Metrics | None = None) -> list[Category]:
        response = self.endpoint.respond(
            message=get_prompt("L1", category=category, all_words=', '.join(
                self.all_words), num_samples=num_samples),
            system_prompt="You are a strategic thinker. Consider the speaker's intentions and possible word combinations.",
            metrics=metrics
        )
        return [
            Category(level=-1, group=category,
                     members=line.strip().split(", "))
            for line in response.strip().split("\n")
        ]


class Speaker:
    def __init__(self, all_words: list[str], endpoint: Endpoint):
        super().__init__()
        self.endpoint = endpoint
        self.all_words = all_words

    def choose_categories(self, words: list[str], num_samples: int = 1) -> list[str]:
        """
        Choose categories for the speaker to communicate given a set of target words.

        :param words: the target words the speaker intends to communicate
        :param num_samples: [optional] the number of samples to generate, defaults to 1
        :return: the name of the chosen category
        """
        raise NotImplementedError


class PragmaticSpeaker(Speaker):

    def __init__(self, all_words: list[str], endpoint: Endpoint, listener: Listener):
        super().__init__(all_words, endpoint)
        self.listener = listener

        if listener.all_words != all_words:
            raise ValueError("listener.all_words must match all_words")

    def choose_categories(self, words: list[str], num_samples: int = 1, metrics: Metrics | None = None) -> list[str]:

        def eval_category(category: str) -> int:
            return self.listener.evaluate_category(category, words, metrics)

        response = self.endpoint.respond(
            message=get_prompt("S1", words=', '.join(
                words), all_words=', '.join(self.all_words)),
            system_prompt="You are a strategic communicator. Choose your words carefully to convey precise meaning.",
            metrics=metrics
        )

        categories = response.strip().split("\n")
        # put the best categories first
        best_categories = sorted(categories, key=eval_category, reverse=True)

        return best_categories[:num_samples]


class RSASolver(Solver):

    @staticmethod
    def _generate_groups(word_bank: list[str], group_size: int = 4) -> Generator[list[str]]:
        """
        Generate all possible groups of words in a word bank 
        given a group size.
        """
        if group_size == 1:
            yield from ([word] for word in word_bank)
            raise StopIteration

        for i, word in enumerate(word_bank[:-group_size+1]):
            yield from [
                [word] + sub_group
                for sub_group in RSASolver._generate_groups(
                    word_bank=word_bank[i+1:],
                    group_size=group_size-1
                )
            ]

    def _evaluate_group(self, word_bank: list[str], proposed_group: list[str], metrics: Metrics | None = None) -> int:
        """
        Evaluate the quality of a given category for describing a set of target
        words by computing the number of target words that are also guessed by
        a literal listener given that category.

        That means 0 is ideal, and higher numbers are worse.

        :param word_bank: the word bank to guess from
        :param proposed_group: the group to evaluate
        :return: the number of target words that are missed by a pragmatic listener
        """
        # TODO: we could try later on with multiple categories or multiple group guesses
        l0 = LiteralListener(
            word_bank,
            ENDPOINTS["literal_listener"]
        )
        s1 = PragmaticSpeaker(
            word_bank,
            ENDPOINTS["speaker"],
            listener=l0
        )
        l1 = PragmaticListener(
            word_bank,
            ENDPOINTS["pragmatic_listener"]
        )

        # Pragmatic Speaker (S1)
        category_utterances: list[str] = s1.choose_categories(
            proposed_group, num_samples=1, metrics=metrics)

        # Pragmatic Listener (L1)
        category_summary = category_utterances[0]
        guesses: list[Category] = l1.guess(category_summary, num_samples=1, metrics=metrics)

        base_category = Category(
            level=-1, group=category_summary, members=proposed_group)

        # Evaluate the guessed sets against the target
        mismatches: list[int] = [
            base_category.diff(guess)
            for guess in guesses
        ]

        return min(mismatches)

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics | None = None) -> tuple[str, ...]:

        error_heap: list[tuple[int, list[str]]] = []
        for proposed_group in RSASolver._generate_groups(word_bank, group_size):
            group_cost = self._evaluate_group(word_bank, proposed_group, metrics)
            heappush(error_heap, (group_cost, proposed_group))

        most_correct_guess = heappop(error_heap)
        cost, words = most_correct_guess
        return tuple(words)
