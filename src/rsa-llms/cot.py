from .endpoints import Endpoint
from .game import Connections, load_json_to_connections, GameOverException

# Define model configuration for CoT prompting
ENDPOINTS = {
    "cot_model": Endpoint("http://localhost:11434", model="gpt-4")  # Adjust the model and endpoint URL as needed
}

def cot_connections_solver(game: Connections, include_category=True, shot_type="zero-shot") -> list[bool]:
    """
    Solve a Connections game using Chain-of-Thought (CoT) prompting, supporting both zero-shot and one-shot modes.

    :param game: The Connections game to solve
    :param include_category: Boolean, whether the agent knows the category it is trying to group
    :param shot_type: Specifies whether to use zero-shot or one-shot prompting ("zero-shot" or "one-shot")
    :return: A list of booleans indicating which levels were solved
    """
    level = 0
    solves = [False, False, False, False]  # Track whether each level is solved

    while level < 4:
        curr_group = game.get_groups_by_level(level)[0]

        # CoT Prompting: Generate reasoning for the category (if known) or based on similarities
        cot_prompt = generate_cot_prompt(curr_group.members, include_category, shot_type)
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
            solves[level] = True
            level += 1
        else:
            # If no correct guess, move on to the next level
            print(f"All guesses failed at level {level}!")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    return solves

def generate_cot_prompt(words: list[str], include_category=True, shot_type="zero-shot") -> str:
    """
    Generate a Chain-of-Thought (CoT) prompt for the given words, with support for both zero-shot and one-shot modes.

    :param words: List of words for the CoT reasoning prompt
    :param include_category: Boolean, whether the agent knows the category it is trying to group
    :param shot_type: Specifies whether to use zero-shot or one-shot prompting ("zero-shot" or "one-shot")
    :return: The generated CoT prompt
    """
    
    # One-shot example when the agent knows the category
    example_with_category = """
Here are some words: apple, banana, orange, grape, car, truck, plane, train.
Group four of these words together and explain what category they belong to.

Example Response:
The words apple, banana, orange, and grape belong to the category of 'fruits.' These are all edible, natural products that grow on trees or vines. The other words are vehicles, which do not belong in this group.
"""

    # One-shot example when the agent does not know the category
    example_without_category = """
Here are some words: apple, banana, orange, grape, car, truck, plane, train.
Group four of these words together based on their similarities, but do not provide the category name.

Example Response:
The words apple, banana, orange, and grape all share a common characteristic: they are types of fruit. They are natural, edible, and grow on trees or vines. The other words are vehicles, so they do not belong in this group.
"""

    # Now provide the actual prompt with the words
    if include_category:
        actual_prompt = f"Here are some words: {', '.join(words)}. Group four of these words together and explain what category they belong to."
    else:
        actual_prompt = f"Here are some words: {', '.join(words)}. Group four of these words together based on their similarities, but do not provide the category name."

    # Return the prompt based on the shot type (zero-shot or one-shot) and include_category
    if shot_type == "one-shot":
        if include_category:
            return example_with_category + actual_prompt
        else:
            return example_without_category + actual_prompt
    else:  # zero-shot
        return actual_prompt

def get_cot_response(prompt: str) -> str:
    """
    Get a response from the CoT model for the provided prompt.

    :param prompt: The CoT reasoning prompt
    :return: The response from the model
    """
    return ENDPOINTS["cot_model"].respond(message=prompt)

def extract_words_from_response(response: str) -> list[str]:
    """
    Extract the group of words from the CoT model's response.

    :param response: The CoT model's response
    :return: A list of four guessed words
    """
    # This is a simple example of extracting words from the response.
    # You might need to adjust this depending on how the model formats its output.
    return response.split()[:4]  # Assuming the first four words are the guessed group

def script_entrypoint():
    # Load the game from the provided ICL connections data
    icl_connections = load_json_to_connections(filename='icl_connections.json')

    # Test the solver with zero-shot and one-shot modes for both include_category options
    print("Starting CoT solver in zero-shot mode where agent knows the category")
    cot_score_zero_shot_with_category = cot_connections_solver(icl_connections, include_category=True, shot_type="zero-shot")
    print(f"CoT solver in zero-shot mode where agent knows the category completed {cot_score_zero_shot_with_category} levels.")

    print("\nStarting CoT solver in one-shot mode where agent knows the category")
    cot_score_one_shot_with_category = cot_connections_solver(icl_connections, include_category=True, shot_type="one-shot")
    print(f"CoT solver in one-shot mode where agent knows the category completed {cot_score_one_shot_with_category} levels.")

    print("\nStarting CoT solver in zero-shot mode where agent does not know the category")
    cot_score_zero_shot_without_category = cot_connections_solver(icl_connections, include_category=False, shot_type="zero-shot")
    print(f"CoT solver in zero-shot mode where agent does not know the category completed {cot_score_zero_shot_without_category} levels.")

    print("\nStarting CoT solver in one-shot mode where agent does not know the category")
    cot_score_one_shot_without_category = cot_connections_solver(icl_connections, include_category=False, shot_type="one-shot")
    print(f"CoT solver in one-shot mode where agent does not know the category completed {cot_score_one_shot_without_category} levels.")

if __name__ == "__main__":
    script_entrypoint()
