from .endpoints import Endpoint
from .game import Connections, load_json_to_connections, GameOverException

# Define model configuration for CoT prompting
ENDPOINTS = {
    "cot_model": Endpoint("http://localhost:11434", model="gpt-4")
}

def cot_connections_solver(game: Connections, include_category=True) -> list[bool]:
    """
    Solve a Connections game using Chain-of-Thought (CoT) prompting.

    :param game: The Connections game to solve
    :param include_category: Boolean, whether to ask the model to provide a category name
    :return: A list of booleans indicating which levels were solved
    """
    level = 0
    solves = [False, False, False, False]  # Track whether each level is solved

    while level < 4:
        curr_group = game.get_groups_by_level(level)[0]

        # CoT Prompting: Generate reasoning for the category
        cot_prompt = generate_cot_prompt(curr_group.members, include_category)
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

def generate_cot_prompt(words: list[str], include_category=True) -> str:
    """
    Generate a Chain-of-Thought (CoT) prompt for the given words.
    
    :param words: List of words for the CoT reasoning prompt
    :param include_category: Boolean, whether to ask for a category name
    :return: The generated CoT prompt
    """
    if include_category:
        prompt = f"Here are some words: {', '.join(words)}. Group four of these words together and explain what category they belong to."
    else:
        prompt = f"Here are some words: {', '.join(words)}. Group four of these words together based on their similarities, but do not provide the category name."
    return prompt

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
    return response.split()[:4]  # Assuming the first four words are the guessed group

def script_entrypoint():
    # Load the game from the provided ICL connections data
    icl_connections = load_json_to_connections(filename='icl_connections.json')

    # Test the solver with and without category names
    print("Starting CoT solver with category names")
    cot_score_with_category = cot_connections_solver(icl_connections, include_category=True)
    print(f"CoT solver with category names completed {cot_score_with_category} levels.")

    print("\nStarting CoT solver without category names")
    cot_score_without_category = cot_connections_solver(icl_connections, include_category=False)
    print(f"CoT solver without category names completed {cot_score_without_category} levels.")

if __name__ == "__main__":
    script_entrypoint()
