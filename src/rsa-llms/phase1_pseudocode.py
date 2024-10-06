
import chevron

from .endpoints import Endpoint
from .game import Connections, Category, load_daily_board, GameOverException

TOTAL_STRIKES = 3
MAX_ITERATIONS = 5

# Define model configurations
MODELS = {
    "speaker": "llama3.2", 
    "pragmatic_listener": "llama3.2",
    "literal_listener": "phi3.5"  # this is a bit simpler version compared to the rest
}

def ollama_call(prompt, system_message="", model="llama3"):
    ollama_endpoint = Endpoint(url="http://localhost:11434", model=model)
    return ollama_endpoint.respond(message=prompt, system_promt=system_message)

def get_prompt(name: str, **kwargs) -> str:
    with open(f"prompts/{name}.mustache") as f:
        return chevron.render(f.read(), **kwargs)

def literal_listener(category: str, all_words: list[str]) -> list[str]:
    prompt = get_prompt("L0", category=category, all_words=', '.join(all_words))

    system_message = "You are a literal interpreter of language. Don't overthink or look for hidden meanings."
    response = ollama_call(prompt, system_message, MODELS["literal_listener"])
    return response.strip().split(", ")

def evaluate_category(category: str, target_words: set[str], all_words: list[str]) -> int:
    """
    Evaluate the quality of a given category for describing a set of target
    words by computing the number of target words that are also guessed by 
    a literal listener given that category.

    In essence, this is P(L_0 interprets target_words out of all_words | category)

    :param category: the category to evaluate
    :return: the number of target words that are also guessed by a literal listener given this category
    """
    literal_guess = set(literal_listener(category, all_words))
    return len(literal_guess.intersection(target_words))

def pragmatic_speaker(words: list[str], all_words: list[str]) -> str:
    target_words = set(words)

    eval_category = lambda category: evaluate_category(category, target_words, all_words)

    prompt = get_prompt("S1", words=', '.join(words), all_words=', '.join(all_words))

    system_message = "You are a strategic communicator. Choose your words carefully to convey precise meaning."
    response = ollama_call(prompt, system_message, MODELS["speaker"])
    
    categories = response.strip().split("\n")
    best_category = max(categories, key=eval_category)
    
    return best_category

def pragmatic_listener(category: str, all_words: list[str], num_samples: int = 1) -> list[Category]:
    prompt = get_prompt("L1", category=category, all_words=', '.join(all_words), num_samples=num_samples)

    system_message = "You are a strategic thinker. Consider the speaker's intentions and possible word combinations."
    response = ollama_call(prompt, system_message, MODELS["pragmatic_listener"])
    return [
        Category(level=-1, group=category, members=line.strip().split(", "))
        for line in response.strip().split("\n")
    ]

def rsa_connections_solver(game: Connections):
    level = 0
    solves = [ False, False, False, False ]

    while level < 4:
        curr_group = game.get_groups_by_level(level)[0]

        # Pragmatic Speaker (S1)
        category_utterance = pragmatic_speaker(curr_group.members, game.all_words)
        print(f"Generated category: {category_utterance}")

        # Pragmatic Listener (L1)
        guesses = pragmatic_listener(category_utterance, game.all_words, num_samples=1) # we could try later on with multiple

        # Check if any of the guessed sets match the target
        for guess in guesses:
            try:
                actual_category = game.guess(guess.members)
            except GameOverException as e:
                raise e

            if actual_category is not None:
                print(f"Level {level} solved! Category: {curr_group.group}")
                solves[level] = True
                level += 1
                break
        else:
            # If no correct guess, use the top guess and move on (to the next level)
            top_guess = guesses[0].members
            print(f"All guesses failed!")
            print(f"Top guess: {', '.join(top_guess)}")
            print(f"Correct: {', '.join(curr_group.members)}")
            level += 1

    return solves

def main():
    daily_board = load_daily_board()

    print("Starting RSA algo")
    rsa_score = rsa_connections_solver(daily_board)
    print(f"RSA-inspired solver completed {rsa_score} levels.")

if __name__ == "__main__":
    main()