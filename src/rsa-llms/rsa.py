
from .game import Category
from .endpoints import get_prompt, Endpoint


class Listener:
    def __init__(self, all_words: list[str], endpoint: Endpoint):
        super().__init__()
        self.endpoint = endpoint
        self.all_words = all_words

    def guess(self, category: str, num_samples: int = 1) -> list[Category]:
        """
        Produce a collection of guesses for the set of target words that the speaker intended to communicate given a category.

        :param category: the category given by the speaker
        :param num_samples: [optional] the number of samples to generate, defaults to 1
        :return: a list of Category objects, each containing a set of target words that the speaker intended
        """
        raise NotImplementedError

    def evaluate_category(self, category: str, target_words: list[str]) -> int:
        """
        Evaluate the quality of a given category for describing a set of target
        words by computing the number of target words that are also guessed by 
        a literal listener given that category.

        In essence, this is P(L interprets target_words out of all_words | category)

        :param category: the category to evaluate
        :return: the number of target words that are also guessed by a literal listener given this category
        """
        guess = set(self.guess(category))
        return len(guess.intersection(set(target_words)))


class LiteralListener(Listener):
    def guess(self, category: str, num_samples: int = 1) -> list[Category]:
        if num_samples != 1:
            raise ValueError("num_samples must be 1 for literal listeners")

        response = self.endpoint.respond(
            message=get_prompt("L0", category=category,
                               all_words=', '.join(self.all_words)),
            system_prompt="You are a literal interpreter of language. Don't overthink or look for hidden meanings."
        )
        return [
            Category(level=-1, group=category,
                     members=response.strip().split(", "))
        ]


class PragmaticListener(Listener):
    def guess(self, category: str, num_samples: int = 1) -> list[Category]:
        response = self.endpoint.respond(
            message=get_prompt("L1", category=category, all_words=', '.join(
                self.all_words), num_samples=num_samples),
            system_prompt="You are a strategic thinker. Consider the speaker's intentions and possible word combinations."
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

    def choose_categories(self, words: list[str], num_samples: int = 1) -> list[str]:

        def eval_category(category: str) -> int:
            return self.listener.evaluate_category(category, words)

        response = self.endpoint.respond(
            message=get_prompt("S1", words=', '.join(
                words), all_words=', '.join(self.all_words)),
            system_prompt="You are a strategic communicator. Choose your words carefully to convey precise meaning.",
        )

        categories = response.strip().split("\n")
        # put the best categories first
        best_categories = sorted(categories, key=eval_category, reverse=True)

        return best_categories[:num_samples]
