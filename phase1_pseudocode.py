import requests
import json
import random

TOTAL_STRIKES = 3
MAX_ITERATIONS = 5

# Define model configurations
MODELS = {
    "speaker": "llama3.2", 
    "pragmatic_listener": "llama3.2",
    "literal_listener": "phi3.5"  # this is a bit simpler version compared to the rest
}

def ollama_call(prompt, system_message="", model="llama3"):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']

# just copied one example
def load_daily_board():
    return {
        "all_words": ["HAIL", "RAIN", "SLEET", "SNOW", "BUCKS", "HEAT", "JAZZ", "NETS", 
                      "OPTION", "RETURN", "SHIFT", "TAB", "KAYAK", "LEVEL", "MOM", "RACECAR"],
        "answers": [
            {"level": 0, "group": "WET WEATHER", "members": ["HAIL", "RAIN", "SLEET", "SNOW"]},
            {"level": 1, "group": "NBA TEAMS", "members": ["BUCKS", "HEAT", "JAZZ", "NETS"]},
            {"level": 2, "group": "KEYBOARD KEYS", "members": ["OPTION", "RETURN", "SHIFT", "TAB"]},
            {"level": 3, "group": "PALINDROMES", "members": ["KAYAK", "LEVEL", "MOM", "RACECAR"]}
        ]
    }

def literal_listener(category, all_words):
    prompt = f"""You are a literal listener in a word game. Given a category, you need to choose words that fit that category exactly as stated, without considering any deeper meanings or connections.

Category: {category}

Choose 4 words from the following list that you think best fit this category literally:
{', '.join(all_words)}

Your response should be only the 4 chosen words, separated by commas."""

    system_message = "You are a literal interpreter of language. Don't overthink or look for hidden meanings."
    response = ollama_call(prompt, system_message, MODELS["literal_listener"])
    return set(response.strip().split(", "))

def pragmatic_speaker(words, all_words):
    target_words = set(words)
    
    def evaluate_category(category):
        literal_guess = literal_listener(category, all_words)
        return len(literal_guess.intersection(target_words))

    prompt = f"""You are a pragmatic speaker in a word game. Your goal is to come up with a category that encompasses these 4 words:
{', '.join(words)}

However, you need to be strategic. The category should be specific enough to include these 4 words, but not so broad that it could include many other words from this list:
{', '.join(all_words)}

Provide 3 different concise category names that you think would lead a listener to choose exactly these 4 words. Each category should be on a new line."""

    system_message = "You are a strategic communicator. Choose your words carefully to convey precise meaning."
    response = ollama_call(prompt, system_message, MODELS["speaker"])
    
    categories = response.strip().split("\n")
    best_category = max(categories, key=evaluate_category)
    
    return best_category

def pragmatic_listener(category, all_words, num_samples=3):
    prompt = f"""You are a pragmatic listener in a word game. Given a category, you need to infer which 4 words the speaker intended you to choose. Consider why the speaker chose this specific category and what they might be trying to communicate.

Category: {category}

Choose 4 words from the following list that you think the speaker intended:
{', '.join(all_words)}

Provide {num_samples} different sets of 4 words each, ordered from most likely to least likely. Each set should be on a new line, with words separated by commas."""

    system_message = "You are a strategic thinker. Consider the speaker's intentions and possible word combinations."
    response = ollama_call(prompt, system_message, MODELS["pragmatic_listener"])
    return [set(line.strip().split(", ")) for line in response.strip().split("\n")]

def rsa_connections_solver(daily_board):
    level = 0
    strikes = TOTAL_STRIKES

    while level < 4 and strikes > 0:
        curr_word_group = daily_board["answers"][level]
        target_words = set(curr_word_group["members"])
        all_words = daily_board["all_words"]

        # Pragmatic Speaker (S1)
        category = pragmatic_speaker(target_words, all_words)
        print(f"Generated category: {category}")

        # Pragmatic Listener (L1)
        guessed_word_sets = pragmatic_listener(category, all_words, num_samples=1) # we could try later on with multiple

        # Check if any of the guessed sets match the target
        for guess_set in guessed_word_sets:
            if guess_set == target_words:
                print(f"Level {level} solved! Category: {curr_word_group['group']}")
                level += 1
                break
        else:
            # If no correct guess, use the top guess
            top_guess = guessed_word_sets[0]
            if top_guess == target_words:
                print(f"Level {level} solved! Category: {curr_word_group['group']}")
                level += 1
            else:
                strikes -= 1
                print(f"Incorrect guess. Strikes remaining: {strikes}")
                print(f"Guessed: {', '.join(top_guess)}")
                print(f"Correct: {', '.join(target_words)}")

    return level

def main():
    daily_board = load_daily_board()

    print("Starting RSA algo")
    rsa_score = rsa_connections_solver(daily_board)
    print(f"RSA-inspired solver completed {rsa_score} levels.")

if __name__ == "__main__":
    main()