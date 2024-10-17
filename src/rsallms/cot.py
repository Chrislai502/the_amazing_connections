import re
from .endpoints import Endpoint, get_prompt
from .game import Connections, load_json_to_connections, GameOverException
from .metrics import Metrics  # Import Metrics

# Define model configuration for CoT prompting
ENDPOINTS = {
    # Adjust the model and endpoint URL as needed
    "cot_model": Endpoint("http://localhost:11434", model="gpt-3.5")
}


def cot_connections_solver(game: Connections, include_category=True, shot_type="zero-shot") -> Metrics:
    """
    Solve a Connections game using Chain-of-Thought (CoT) prompting, supporting both zero-shot and one-shot modes.

    :param game: The Connections game to solve
    :param include_category: Boolean, whether the agent knows the category it is trying to group
    :param shot_type: Specifies whether to use zero-shot or one-shot prompting ("zero-shot" or "one-shot")
    :return: A Metrics object containing the results
    """
    metrics = Metrics(total_levels=4)
    level = 0

    while level < metrics.total_levels:
        curr_group = game.get_groups_by_level(level)[0]

        # CoT Prompting: Generate reasoning for the category (if known) or based on similarities
        if include_category:
            cot_prompt = generate_cot_prompt(
                curr_group.members, include_category, shot_type, category=curr_group.group)
        else:
            cot_prompt = generate_cot_prompt(
                curr_group.members, include_category, shot_type)

        category_utterance = get_cot_response(cot_prompt)

        print(f"Generated category reasoning: {category_utterance}")

        # CoT Guessing: Extract guessed words from CoT response
        guess = extract_words_from_response(category_utterance)

        # Check if the guessed set matches the target
        try:
            actual_category = game.guess(guess)
        except GameOverException as e:
            raise e

        if actual_category is not None:
            print(f"Level {level} solved! Category: {curr_group.group}")
            metrics.add_solve(level)
            level += 1
        else:
            metrics.increment_failed_guesses()
            print(f"All guesses failed at level {level}!")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    metrics.finalize_points()
    return metrics


def generate_cot_prompt(words: list[str], include_category=True, shot_type="zero-shot", category=None) -> str:
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

    if include_category:
        if category is None:
            raise ValueError(
                "Category must be provided when include_category=True")

        # Use the template for including category
        data = {
            "words": words_str,
            "category": category
        }
        prompt = get_prompt("naive_with_category", **data)

        # Load one-shot example from Mustache file
        one_shot_example = get_prompt("one_shot_with_category")

    else:
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


def get_cot_response(prompt: str) -> str:
    """
    Simulated response from the CoT model for the provided prompt. This is useful for testing.
    """
    print(f"Prompt sent to model:\n{prompt}\n")

    # Simulated model response
    if "category" in prompt:
        return "Apple, banana, orange, and grape belong to the category of 'fruits.' These are all edible, natural products."
    else:
        return "Apple, banana, orange, and grape all share a common characteristic: they are types of fruit."


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
        filename='src/rsa-llms/icl_connections.json')
    game_to_solve = icl_connections[-1]

    # Test the solver with zero-shot mode where agent knows the category
    print("Starting CoT solver in zero-shot mode where agent knows the category")
    metrics_zero_shot_with_category = cot_connections_solver(
        game_to_solve, include_category=True, shot_type="zero-shot")
    print(f"Metrics: {metrics_zero_shot_with_category.to_dict()}")

    # Test the solver with one-shot mode where agent knows the category
    print("\nStarting CoT solver in one-shot mode where agent knows the category")
    game_to_solve.reset()  # Reset the game state
    metrics_one_shot_with_category = cot_connections_solver(
        game_to_solve, include_category=True, shot_type="one-shot")
    print(f"Metrics: {metrics_one_shot_with_category.to_dict()}")

    # Test the solver with zero-shot mode where agent does not know the category
    print("\nStarting CoT solver in zero-shot mode where agent does not know the category")
    game_to_solve.reset()  # Reset the game state
    metrics_zero_shot_without_category = cot_connections_solver(
        game_to_solve, include_category=False, shot_type="zero-shot")
    print(f"Metrics: {metrics_zero_shot_without_category.to_dict()}")

    # Test the solver with one-shot mode where agent does not know the category
    print("\nStarting CoT solver in one-shot mode where agent does not know the category")
    game_to_solve.reset()  # Reset the game state
    metrics_one_shot_without_category = cot_connections_solver(
        game_to_solve, include_category=False, shot_type="one-shot")
    print(f"Metrics: {metrics_one_shot_without_category.to_dict()}")


if __name__ == "__main__":
    script_entrypoint()
