from enum import Enum, auto
from typing import List
from rsallms.metrics import Metrics
import random
import re
import json
from .rsallms import CustomModelClient, Connections

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


class State(Enum):
    INITIALIZATION = auto()
    CATEGORY_GENERATION = auto()
    WORD_PREDICTION = auto()
    EVALUATION = auto()
    TERMINATION = auto()


class StateFlowGame:
    def __init__(self, game: Connections):
        self.game = game
        self.metrics = Metrics()
        self.state = State.INITIALIZATION
        self.strikes = 0
        self.max_strikes = 3
        self.all_words = game.all_words.copy()
        self.remaining_words = self.all_words.copy()
        self.solved_categories = []
        self.current_candidate_words = []
        self.current_category = ''
        self.current_category_level = 0

        # LLM configuration
        config_list_custom = autogen.config_list_from_json(
            env_or_file="autogen_agents.json",
            filter_dict={
                "model_client_cls": ["CustomModelClient"]
            },
        )
        self.llm_config = {"config_list": config_list_custom}
        # self.llm_config = {
        #     'model': 'gpt-3.5-turbo',
        #     'temperature': 0.7
        # }

        # Initialize agents
        self.alice_agent = AssistantAgent(
            name='Alice',
            system_message="""You are Alice, a language model that generates a category for the given words.
Given the following words, provide a concise category that encompasses these words.""",
            llm_config=self.llm_config
        )
        self.alice_agent.register_model_client(
            model_client_cls=CustomModelClient)

        self.bob_agent = AssistantAgent(
            name='Bob',
            system_message="""You are Bob, a language model that selects words fitting a given category.
Given a category and a list of words, select exactly 4 words that best fit the category.
Provide your answer in the format: ["word1", "word2", "word3", "word4"]""",
            llm_config=self.llm_config
        )
        self.bob_agent.register_model_client(
            model_client_cls=CustomModelClient)

        self.evaluator_agent = AssistantAgent(
            name='Evaluator',
            system_message="""You are the Evaluator. Compare Bob's predicted words with the ground truth.
Determine if Bob's predictions are correct. If correct, proceed to update the game state.
If incorrect, increment the strike count. If strikes reach 3, terminate the game.""",
            llm_config=self.llm_config
        )
        self.evaluator_agent.register_model_client(
            model_client_cls=CustomModelClient)

        self.groupchat = GroupChat(
            agents=[self.alice_agent, self.bob_agent, self.evaluator_agent],
            messages=[],
            max_round=100,
            speaker_selection_method=self.state_transition
        )

        self.manager = GroupChatManager(
            groupchat=self.groupchat,
            llm_config={'temperature': 0.0}
        )

    def run(self):
        self.manager.reset_chat()

        while self.state != State.TERMINATION:
            if self.state == State.INITIALIZATION:
                self.initialize()
            elif self.state == State.CATEGORY_GENERATION:
                self.category_generation()
            elif self.state == State.WORD_PREDICTION:
                self.word_prediction()
            elif self.state == State.EVALUATION:
                self.evaluation()
        print(f"Game completed with {self.strikes} strikes.")
        print(f"Metrics: {self.metrics.to_dict()}")

    def initialize(self):
        self.state = State.CATEGORY_GENERATION

    def category_generation(self):
        unsolved_categories = [cat for cat in self.game.categories]
        if not unsolved_categories:
            self.state = State.TERMINATION
            return
        selected_category = random.choice(unsolved_categories)
        self.current_candidate_words = selected_category.members
        self.current_category_level = selected_category.level

        alice_prompt = f"Words: {', '.join(self.current_candidate_words)}\nPlease provide a concise category that encompasses these words."

        # Alice generates a category
        alice_response = self.alice_agent.complete(alice_prompt)
        self.current_category = alice_response.strip()
        print(f"Alice's Category: {self.current_category}")
        self.state = State.WORD_PREDICTION

    def word_prediction(self):
        # Bob predicts words based on the category and the full set of words
        bob_prompt = f"Category: {self.current_category}\nWords: {', '.join(self.remaining_words)}\nPlease select exactly 4 words from the list that best fit the category.\nProvide your answer in the format: [\"word1\", \"word2\", \"word3\", \"word4\"]"
        bob_response = self.bob_agent.complete(bob_prompt)
        self.bob_predicted_words = self.parse_bob_response(bob_response)
        print(f"Bob's Predicted Words: {self.bob_predicted_words}")
        self.state = State.EVALUATION

    def parse_bob_response(self, response: str) -> List[str]:
        # Parse Bob's response to extract the words
        try:
            # Try to parse the response as a JSON array
            words = json.loads(response)
            if isinstance(words, list):
                words = [word.strip().upper() for word in words if word.strip().upper() in [
                    w.upper() for w in self.all_words]]
                return words[:4]
        except json.JSONDecodeError:
            pass
        # If JSON parsing fails, use regex
        match = re.findall(r'["\']?(\w+)["\']?', response)
        words = [word.strip().upper() for word in match if word.strip().upper() in [
            w.upper() for w in self.all_words]]
        return words[:4]  # Limit to 4 words

    def evaluation(self):
        # Evaluate Bob's predictions
        correct = set(self.bob_predicted_words) == set(
            [word.upper() for word in self.current_candidate_words])
        if correct:
            # Correct
            print(f"Correctly identified category: {self.current_category}")
            # Update metrics
            self.metrics.add_solve(self.current_category_level)
            # Update category similarity metric
            correct_category = next((cat for cat in self.game._og_groups if set(
                cat.members) == set(self.current_candidate_words)), None)
            if correct_category:
                self.metrics.cosine_similarity_category(
                    self.current_category, correct_category.group)
            # Remove the correctly identified words from remaining words
            for word in self.current_candidate_words:
                if word in self.remaining_words:
                    self.remaining_words.remove(word)
            # Remove the category from the game's categories
            self.game.categories = [
                cat for cat in self.game.categories if cat.members != self.current_candidate_words]
            if not self.remaining_words or not self.game.categories:
                self.state = State.TERMINATION
            else:
                self.state = State.CATEGORY_GENERATION
        else:
            # Incorrect
            print(f"Incorrect guess. Bob's predicted words: {self.bob_predicted_words}")
            self.strikes += 1
            self.metrics.increment_failed_guesses()
            # Check for hallucinations
            hallucinated_words = self.metrics.hallucination_words(
                self.bob_predicted_words, self.all_words)
            print(f"Hallucinated Words: {hallucinated_words}")
            if self.strikes >= self.max_strikes:
                self.state = State.TERMINATION
            else:
                self.state = State.CATEGORY_GENERATION

    def state_transition(self, last_speaker, groupchat):
        # Custom speaker selection function
        if self.state == State.CATEGORY_GENERATION:
            return self.alice_agent
        elif self.state == State.WORD_PREDICTION:
            return self.bob_agent
        elif self.state == State.EVALUATION:
            return self.evaluator_agent
        else:
            return None  # Terminate the chat
