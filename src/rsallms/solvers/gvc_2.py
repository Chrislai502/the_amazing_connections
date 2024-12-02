from .solver import Solver
from ..game import Connections, GameOverException
from ..metrics import Metrics
# from ..endpoints import ENDPOINTS
import os

from autogen import ConversableAgent, GroupChat, GroupChatManager, Agent

class GVCSolver(Solver):
    def __init__(self):
        super().__init__()
        self.initialize_agents()
    
    def initialize_agents(self):

        self.llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

        # Create agents using ConversableAgent
        self.guesser_agent = ConversableAgent(
            name="GuesserAgent",
            system_message="You are a Guesser Agent in a word game. Given a list of words, propose a group of 4 related words and a corresponding category.",
            llm_config=self.llm_config,
        )

        self.validator_agent = ConversableAgent(
            name="ValidatorAgent",
            system_message="You are a Validator Agent. Given a list of words and a category, identify a group of 4 words that fit the category.",
            llm_config=self.llm_config,
        )

        self.consensus_agent = ConversableAgent(
            name="ConsensusAgent",
            system_message="You are a Consensus Agent. Compare two groups of words and determine if they match.",
            llm_config=self.llm_config,
        )

        # Register agent behaviors
        self.register_agent_behaviors()

        # Define allowed speaker transitions
        self.allowed_transitions = {
            self.guesser_agent: [self.validator_agent],
            self.validator_agent: [self.consensus_agent],
            self.consensus_agent: [self.guesser_agent],  # For feedback loop
        }

    def register_agent_behaviors(self):
        # Guesser Agent's reply function
        def guesser_response(agent, message, sender, config):
            words = message["content"]
            feedback = agent.memory.get('feedback', '')
            prompt = f"""{feedback}
                        Given the following words: {words}
                        Find a group of 4 related words and provide a category.
                        Format:
                        Group: word1, word2, word3, word4
                        Category: category_name"""
            response = agent.llm_completion(prompt)
            agent.memory['feedback'] = ''
            return True, response

        def validator_response(agent, message, sender, config):
            content = message["content"]
            lines = content.strip().split('\n')
            group_line = next((line for line in lines if line.startswith('Group:')), '')
            category_line = next((line for line in lines if line.startswith('Category:')), '')
            words = ', '.join(self.game.all_words)  # Use the full game board
            category = category_line.replace('Category:', '').strip()
            prompt = f"""Given the following words: {words}
                        And the category: {category}
                        Identify the 4 words that belong to this category.
                        Format:
                        Group: word1, word2, word3, word4"""
            response = agent.llm_completion(prompt)
            return True, response

        # Consensus Agent's reply function
        def consensus_response(agent, message, sender, config):
            content = message["content"]
            lines = content.strip().split('\n')
            guesser_group_line = next((line for line in lines if line.startswith('Guesser Group:')), '')
            validator_group_line = next((line for line in lines if line.startswith('Validator Group:')), '')
            guesser_group = set(guesser_group_line.replace('Guesser Group:', '').strip().split(', '))
            validator_group = set(validator_group_line.replace('Validator Group:', '').strip().split(', '))
            if guesser_group == validator_group:
                return True, "Consensus reached. The groups match. Submitting the guess."
            else:
                feedback = "The Validator Agent identified a different group for the category. Please reconsider your guess."
                # Store feedback in Guesser Agent's memory
                self.guesser_agent.memory['feedback'] = feedback
                return True, feedback

        # Register reply functions using register_reply
        self.guesser_agent.register_reply(
            trigger=[Agent, None],  # Trigger on any message
            reply_func=guesser_response
        )
        self.validator_agent.register_reply(
            trigger=[Agent, None],
            reply_func=validator_response
        )
        self.consensus_agent.register_reply(
            trigger=[Agent, None],
            reply_func=consensus_response
        )

    def guess(self, word_bank: list[str], group_size: int = 4, previous_guesses: set[tuple[str, ...]] = set(), metrics: Metrics = None, history: str = "") -> tuple[tuple[str, ...], str]:
        """
        Implement the GVC framework to make a guess.

        :param word_bank: All of the words remaining in the game.
        :param group_size: Number of words to guess as a group (default: 4).
        :param previous_guesses: Set of previous guesses (unused here).
        :param metrics: Metrics object for tracking (optional).
        :param history: History of previous interactions (unused here).
        :return: A tuple containing the guessed words and reasoning.
        """
        
        # Prepare the game board as a string
        game_board_str = ', '.join(word_bank)
        initial_message = f"Words: {game_board_str}"

        # Set up the GroupChat
        agents = [self.guesser_agent, self.validator_agent, self.consensus_agent]

        group_chat = GroupChat(
            agents=agents,
            allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
            speaker_transitions_type="allowed",
            messages=[],
            max_round=10,
            send_introductions=True,
        )

        # Configure GroupChatManager according to documentation
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
            speaker_selection_method="auto",
            select_speaker_message_template="You are coordinating agents in a word game. The agents are: {roles}. Based on the conversation, select the next agent to speak from {agentlist}. Only return the agent's name.",
            select_speaker_prompt_template="Based on the above conversation, who should speak next? Choose from {agentlist}. Only return the agent's name.",
            role_for_select_speaker_messages="system",
            max_retries_for_selecting_speaker=2,
        )

        # Initialize chat history for the agents
        self.guesser_agent.chat_messages = {}
        self.validator_agent.chat_messages = {}
        self.consensus_agent.chat_messages = {}
        group_chat_manager.chat_messages = {}

        # Start the conversation
        chat_result = self.guesser_agent.initiate_chat(
            recipient=group_chat_manager,
            message=initial_message,
            max_turns=10,
            summary_method="none",
        )

        # Extract the final guess from the chat history
        guess = []
        reasoning = ""
        for message in chat_result.chat_history:
            if message['sender'] == 'ConsensusAgent' and 'Consensus reached' in message['content']:
                # Find the Guesser Agent's last message
                for prior_msg in reversed(chat_result.chat_history):
                    if prior_msg['sender'] == 'GuesserAgent' and 'Group:' in prior_msg['content']:
                        lines = prior_msg['content'].strip().split('\n')
                        group_line = next((line for line in lines if line.startswith('Group:')), '')
                        category_line = next((line for line in lines if line.startswith('Category:')), '')
                        guess = group_line.replace('Group:', '').strip().split(', ')
                        reasoning = category_line.replace('Category:', '').strip()
                        break
                break

        if not guess:
            # Fallback if no consensus was reached
            raise ValueError("No consensus reached. Unable to make a guess.")

        return tuple(guess), reasoning

    def play(self, game: Connections, commit_to: str | None = None) -> list[bool]:
        """
        Play the game using the GVCSolver.

        :param game: The Connections game instance.
        :param commit_to: Optional database to commit metrics.
        :return: List indicating which categories were solved.
        """
        self.game = game  # Make the game accessible to agents
        metrics = Metrics()
        previous_guesses: set[tuple[str, ...]] = set()
        history: str = ""

        while not game.is_over:
            try:
                guess, reasoning = self.guess(
                    word_bank=game.all_words,
                    group_size=game.group_size,
                    previous_guesses=previous_guesses,
                    metrics=metrics,
                    history=history
                )
                # Attempt to check the guess
                cat = game.category_guess_check(list(guess))
                print(f"Guessed: {guess} --> {cat}")

                if cat is None:
                    previous_guesses.add(tuple(guess))
                    metrics.hallucination_words(list(guess), game.all_words)
                    metrics.increment_failed_guesses()
                    if history == "":
                        history += "History: "
                    history += f"Failed Guess: Word Grouping: {guess} Reasoning: {reasoning}\n"
                else:
                    guessed_cat_idx = game._og_groups.index(cat)
                    metrics.add_solve(level=guessed_cat_idx)
                    metrics.cosine_similarity_category(guessed_cat=reasoning, correct_cat=cat.group)
                    # Remove guessed words from the word bank
                    for word in guess:
                        if word in game.all_words:
                            game.all_words.remove(word)
            except GameOverException as e:
                print(str(e))
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        if commit_to is not None:
            metrics.commit(to_db=commit_to)
        return game.solved_categories
