
import requests, random
from dataclasses import dataclass

# the repository for this data is at https://github.com/Eyefyre/NYT-Connections-Answers
GAME_DATA_ENDPOINT = "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/refs/heads/main/connections.json"

class GameOverException(Exception): pass

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

class Connections:

    @property
    def all_words(self) -> list[str]:
        return [
            word 
            for group in self.groups
            for word in group.members
        ]

    def __init__(self, groups: list[Category], group_size: int = 4, max_strikes: int = 9999):
        """
        Initialize a Connections object with a list of groups and their associated members.

        :param groups: a list of Category objects
        :param group_size: the size of each group (default: 4)

        :raises ValueError: if not all groups have the size specified by group_size
        """
        # Check that all groups have the same size
        if not all(len(group.members) == group_size for group in groups):
            raise ValueError(f"All groups must have exactly {group_size} members")

        self._max_strikes = max_strikes
        self._og_groups = groups
        self.groups = groups
        self.current_strikes = 0

    def get_groups_by_level(self, level: int) -> list[Category]:
        """Filter the groups in this game by their level"""
        return [group for group in self.groups if group.level == level]

    def guess(self, words: list[str]) -> Category | None:
        """
        Returns the category associated with the given words if they match any of the groups
        (and removes that category from the list), otherwise returns None.
        """

        if self.current_strikes >= self._max_strikes:
            raise GameOverException("Game over. You've reached the max number of strikes!")

        matches = [
            group.matches(words) for group in self.groups
        ]

        good_guess = any(matches)

        if not good_guess:
            self.current_strikes += 1
            return None
        
        matched_group = matches.index(True)
        return self.groups.pop(matched_group)
    
    def reset(self):
        self.groups = self._og_groups
        self.current_strikes = 0

def sample_game() -> Connections:
    """
    Returns a Connections object randomly sampled from historical game data.
    """
    resp = requests.get(GAME_DATA_ENDPOINT)
    if resp.status_code != 200:
        raise Exception(f"Failed to get connections data: {resp.status_code}")
    games = resp.json()

    sampled_game = random.sample(games, 1)

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

    categories = [ ]
    for game in games:
        categories.extend(game["answers"])
    
    categories = [ Category(**category) for category in categories ]

    sampled_categories = random.sample(categories, 4)

    return Connections(sampled_categories)


# just copied one example
def load_daily_board() -> Connections:
    """
    FOR TESTING. REMOVE ME!
    """
    return Connections([
        Category(level=0, group="WET WEATHER", members=["HAIL", "RAIN", "SLEET", "SNOW"]),
        Category(level=1, group="NBA TEAMS", members=["BUCKS", "HEAT", "JAZZ", "NETS"]),
        Category(level=2, group="KEYBOARD KEYS", members=["OPTION", "RETURN", "SHIFT", "TAB"]),
        Category(level=3, group="PALINDROMES", members=["KAYAK", "LEVEL", "MOM", "RACECAR"])
    ])
