# game.py

import random
from dataclasses import dataclass, asdict
import requests
import json

from .metrics import Metrics

# the repository for this data is at https://github.com/Eyefyre/NYT-Connections-Answers
GAME_DATA_ENDPOINT = "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/refs/heads/main/connections.json"


class GameOverException(Exception):
    pass


@dataclass
class Category:
    """
    Schema for a category in the connections.json file
    """
    level: int
    group: str
    members: list[str]

    def matches(self, words: list[str]) -> bool:
        return set(words) == set(self.members)

    def diff(self, other_category: "Category") -> int:
        """
        Get the number of words mismatching between two categories
        """
        this_set = set(self.members)
        other_set = set(other_category.members)
        return len(this_set.symmetric_difference(other_set))


class Connections:
    """
    A single game of Connections
    """

    @property
    def all_words(self) -> list[str]:
        return [
            word
            for group in self.categories
            for word in group.members
        ]

    @property
    def get_words_per_group(self) -> list[dict[str, int | str | list[str]]]:
        """
        The categories in this game as a list of dictionaries
        """
        return [asdict(group) for group in self.categories]

    @property
    def is_solved(self) -> bool:
        """
        A game is solved if all categories have been guessed
        """
        return len(self.categories) == 0

    @property
    def is_over(self) -> bool:
        """
        A game is over if the max number of strikes is reached or
        if the game is solved
        """
        return (
            self.current_strikes >= self._max_strikes
            or self.is_solved
        )

    @property
    def solved_categories(self) -> list[bool]:
        """
        Produce a list of flags indicating which categories have
        been guessed
        """
        return [
            not (cat in self.categories)
            for cat in self._og_groups
        ]

    def __init__(self, categories: list[Category], group_size: int = 4, max_strikes: int = 9999, starting_strikes: int = 0):
        """
        Initialize a Connections object with a list of categories and
        their associated members.

        :param categories: a list of Category objects
        :param group_size: the size of each group (default: 4)

        :raises ValueError: if not all groups have the size specified
        by group_size
        """
        # Check that all groups have the same size
        if not all(len(group.members) == group_size for group in categories):
            raise ValueError(f"All groups must have exactly {group_size} members")

        self._metrics = Metrics(total_levels=len(categories))
        self._max_strikes = max_strikes
        self._og_groups = categories.copy()
        self.group_size = group_size
        self.categories = categories.copy()
        self.current_strikes = starting_strikes

    def get_metrics(self) -> Metrics:
        self._metrics.finalize_points()
        return self._metrics

    def get_groups_by_level(self, level: int) -> list[Category]:
        """Filter the groups in this game by their level"""
        return [group for group in self.categories if group.level == level]

    def json(self) -> dict[str, list[dict[str, int | str | list[str]]] | int]:
        return {
            "groups": [asdict(g) for g in self.categories],
            "group_size": len(self.categories[0].members),
            "max_strikes": self._max_strikes,
            "starting_strikes": self.current_strikes
        }

    def guess(self, words: list[str]) -> Category | None:
        """
        Return the category associated with the guessed words if they match a category
        (and removes that category from the game), otherwise return None and add a strike
        """
        if self.current_strikes >= self._max_strikes:
            raise GameOverException(
                "Game over. You've reached the max number of strikes!")

        matches = [
            group.matches(words) for group in self.categories
        ]

        good_guess = any(matches)

        if not good_guess:
            self.current_strikes += 1
            self._metrics.increment_failed_guesses()
            return None

        matched_group = matches.index(True)
        solved_category = self.categories.pop(matched_group)
        self._metrics.add_solve(matches.index(True))
        return solved_category

    def reset(self):
        """
        Reset the game to its initial state
        """
        self._metrics = Metrics(total_levels=len(self._og_groups))
        self.categories = self._og_groups.copy()
        self.current_strikes = 0


def sample_game() -> Connections:
    """
    Returns a Connections object randomly sampled from historical game data.
    """
    resp = requests.get(GAME_DATA_ENDPOINT)
    if resp.status_code != 200:
        raise Exception(f"Failed to get connections data: {resp.status_code}")
    games = resp.json()

    sampled_game = random.sample(games, 1)[0]

    categories = [
        Category(**category) for category in sampled_game["answers"]
    ]

    return Connections(categories)


def mixed_game() -> Connections:
    """
    Returns a Connections object with a sample of 4 categories from historical game data.

    Note: The resulting game may or may not have been a historical game.
    """
    resp = requests.get(GAME_DATA_ENDPOINT)
    if resp.status_code != 200:
        raise Exception(f"Failed to get connections data: {resp.status_code}")
    games = resp.json()

    categories = []
    for game in games:
        categories.extend(game["answers"])

    categories = [Category(**category) for category in categories]

    sampled_categories = random.sample(categories, 4)

    return Connections(sampled_categories)


# just copied one example
def load_daily_board() -> Connections:
    """
    FOR TESTING. REMOVE ME!
    """
    return Connections([
        Category(level=0, group="WET WEATHER", members=[
                 "HAIL", "RAIN", "SLEET", "SNOW"]),
        Category(level=1, group="NBA TEAMS", members=[
                 "BUCKS", "HEAT", "JAZZ", "NETS"]),
        Category(level=2, group="KEYBOARD KEYS", members=[
                 "OPTION", "RETURN", "SHIFT", "TAB"]),
        Category(level=3, group="PALINDROMES", members=[
                 "KAYAK", "LEVEL", "MOM", "RACECAR"])
    ])


def load_games() -> list[Connections]:
    """Load all games from the remote endpoint."""
    resp = requests.get(GAME_DATA_ENDPOINT)
    if resp.status_code != 200:
        raise Exception(f"Failed to get connections data: {resp.status_code}")

    raw_data = resp.json()

    if not isinstance(raw_data, list):
        raise ValueError(f"Games data is not a list of games!")

    return [
        Connections(categories=[
            Category(**cat)
            for cat in game["answers"]
        ]) for game in raw_data
    ]


def save_specific_game_indices_to_json(indices: list[int], filename='connections.json') -> None:
    """Save the specified game connections to a JSON file."""
    games: list[Connections] = load_games()
    categories: list[Category] = []

    for idx in indices:
        if idx < len(games):
            game: Connections = games[idx]
            # Collect categories from this game
            categories.extend(game._og_groups)
        else:
            raise IndexError(f"Index {idx} is out of range!")

    # Save the categories to JSON
    with open(filename, 'w') as f:
        json.dump([asdict(cat) for cat in categories], f)


def load_json_to_connections(filename: str) -> list[Connections]:
    """Load list of categories from a JSON file into list of Connections games."""
    with open(filename, 'r') as f:
        data = json.load(f)

    categories = [Category(**item) for item in data]
    return [Connections(categories[i:i+4]) for i in range(0, len(categories), 4)]


if __name__ == "__main__":
    icl_indices = [218, 348, 390, 64, 158, 197, 77, 401, 219, 284]
    test_indices = [469, 4, 113, 466, 39, 301, 312, 254, 15,
                    239, 204, 149, 209, 25, 276, 132, 208, 428, 272, 142]

    # Save the specific game connections to a JSON file
    save_specific_game_indices_to_json(
        icl_indices, filename='icl_connections.json')

    # Load the connections from the saved JSON file
    icl_connections = load_json_to_connections('icl_connections.json')

    # Save the specific game connections to a JSON file
    save_specific_game_indices_to_json(
        test_indices, filename='test_connections.json')

    # Load the connections from the saved JSON file
    test_connections = load_json_to_connections('test_connections.json')
    print(test_connections[12].get_words_per_group)
