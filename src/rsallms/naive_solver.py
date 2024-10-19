import re
from .endpoints import Endpoint, get_prompt
from .game import Connections, load_json_to_connections, GameOverException
from .metrics import Metrics  # Import Metrics
import random

# # Define model configuration for CoT prompting
# ENDPOINTS = {
#     # Adjust the model and endpoint URL as needed
#     "naive": Endpoint("http://localhost:11434", model="llama3.2")
# }


def naive_connections_solver(game: Connections, include_category=True, shot_type="zero-shot") -> Metrics:
    """
    Solve a Connections game with simple prompting.

    :param game: The Connections game to solve
    :param include_category: Boolean, whether the agent knows the category it is trying to group
    :param shot_type: Specifies whether to use zero-shot or one-shot prompting ("zero-shot" or "one-shot")
    :return: A Metrics object containing the results
    """
    level = 0
    metrics = Metrics(total_levels=4)
    board = game.get_words_per_group
    board_words = []
    for i in range(Metrics.total_levels):
        words = board[i]['members']
        if isinstance(words, list):
            for w in words:
                board_words += [w]
    random.seed(42)
    random.shuffle(board_words)
    system_prompt = get_prompt("system")
    while level < metrics.total_levels:
        curr_group = game.get_groups_by_level(level)[0]

        if include_category:
            naive_prompt = generate_naive_prompt(
                board_words, include_category, shot_type, category=curr_group.group)
        else:
            naive_prompt = generate_naive_prompt(
                board_words, include_category, shot_type
            )
        category_utterance = get_naive_response(naive_prompt, system_prompt=system_prompt)

        print(f"Generated category reasoning: {category_utterance}")

        guess = extract_words_from_response(category_utterance) # set of 4 words guessed
        # Check if the guessed set matches the target
        try:
            actual_category = game.guess(guess)
        except GameOverException as e:
            raise e
        if actual_category is not None:
            print(f"Level {level} solved! Category: {curr_group.group}")
            for g in guess:
                board_words.remove(g)
            metrics.add_solve(level)
            level += 1
        else:
            metrics.increment_failed_guesses()
            print(f"All guesses failed at level {level}!")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    metrics.finalize_points()
    return metrics

# TODO: Need to implement 
def get_naive_response(prompt: str, system_prompt: str) -> str:
    """
    Simulated response from the CoT model for the provided prompt. This is useful for testing.
    """
    if not endpoint_set:
        # Simulated model response
        if "category" in prompt:
            return "CRICKET, FROG, HARE, KANGAROO belong to the category of 'JUMPING ANIMALS.' These are all edible, natural products."
        else:
            return "CRICKET, FROG, HARE, KANGAROO all share a common characteristic: they are types of JUMPING ANIMALS."
    else:
        print(f"Prompt sent to model:\n{prompt}\n")
        return endpoint.respond(message=prompt, system_prompt=system_prompt)



def generate_naive_prompt(words: list[str], include_category=True, shot_type="zero-shot", category=None) -> str:
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
    num_words = len(words)

    if include_category:
        if category is None:
            raise ValueError(
                "Category must be provided when include_category=True")

        # Use the template for including category
        data = {
            "words": words_str,
            "category": category,
            "num_words": num_words,
        }
        prompt = get_prompt("zero_shot_with_category", **data)

    else:
        # Use the template for not including category
        data = {
            "words": words_str,
            "num_words": num_words,
        }
        prompt = get_prompt("zero_shot_without_category", **data)

    return prompt

def extract_words_from_response(response: str) -> list[str]:
    """
    Extract the group of words from the CoT model's response.

    :param response: The CoT model's response
    :return: A list of four guessed words
    """
    # Regular expression to find words between the phrase "belong to" or "are" and "category" or the end of the sentence.
    # This assumes the model outputs something like: "Apple, banana, orange, and grape belong to the category..."
    pattern = r"\b[A-Za-z]+\b"

    # Find all words in the response
    words = re.findall(pattern, response)

    # Return the first four words
    return words[:4]

def script_entrypoint():
    # Load the game from the provided ICL connections data
    icl_connections = load_json_to_connections(
        filename='src/rsallms/icl_connections.json')
    game_to_solve = icl_connections[-1]

    # Test the solver with zero-shot mode where agent knows the category
    print("Starting Naive solver in zero-shot mode where agent knows the category")
    metrics_zero_shot_with_category = naive_connections_solver(
        game_to_solve, include_category=True, shot_type="zero-shot")
    print(f"Metrics: {metrics_zero_shot_with_category.to_dict()}")

    # Test the solver with zero-shot mode where agent does not know the category
    print("\nStarting Naive solver in zero-shot mode where agent does not know the category")
    game_to_solve.reset()  # Reset the game state
    metrics_zero_shot_without_category = naive_connections_solver(
        game_to_solve, include_category=False, shot_type="zero-shot")
    print(f"Metrics: {metrics_zero_shot_without_category.to_dict()}")


if __name__ == "__main__":
    endpoint_set = False
    if not True:
        endpoint = Endpoint("groq", "gpt3.5", "api_key")
        endpoint_set = True
    script_entrypoint()
