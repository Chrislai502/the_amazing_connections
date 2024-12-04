import os
import logging
from typing import Optional, Dict, Any, List, Tuple, Set

from .solver import Solver
from ..game import Connections, GameOverException
from ..metrics import Metrics

from autogen import ConversableAgent, Agent

# Configure logging at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GVCSolver(Solver):
    def __init__(self):
        super().__init__()
        # Initialize external memory for agents
        self.agent_memory: Dict[str, Dict[str, Any]] = {}
        self.initialize_agents()

    def initialize_agents(self):
        """
        Initialize ConversableAgents and set up their configurations.
        """
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY")
            }]
        }

        # Create agents using ConversableAgent
        self.guesser_agent = ConversableAgent(
            name="GuesserAgent",
            system_message=(
                "You are a Guesser Agent in a word game. Given a list of words, "
                "propose a group of 4 related words and a corresponding category."
            ),
            llm_config=self.llm_config,
        )

        self.validator_agent = ConversableAgent(
            name="ValidatorAgent",
            system_message=(
                "You are a Validator Agent. Given a list of words and a category, "
                "identify a group of 4 words that fit the category."
            ),
            llm_config=self.llm_config,
        )

        self.consensus_agent = ConversableAgent(
            name="ConsensusAgent",
            system_message=(
                "You are a Consensus Agent. Compare two groups of words and determine if they contain exactly the same words, regardless of their order. Respond with 'Consensus reached' if both groups have identical words, or 'Consensus not reached' otherwise."
            ),
            llm_config=self.llm_config,
        )

        # Initialize memory for each agent
        for agent in [self.guesser_agent, self.validator_agent, self.consensus_agent]:
            self.agent_memory.setdefault(agent.name, {})

    def guess(
        self, 
        word_bank: List[str], 
        group_size: int = 4, 
        previous_guesses: Set[Tuple[str, ...]] = set(), 
        metrics: Optional[Metrics] = None, 
        history: str = ""
    ) -> Tuple[Tuple[str, ...], str]:
        """
        Implement the fixed interaction flow to make a guess, including retries if necessary.

        :param word_bank: All of the words remaining in the game.
        :param group_size: Number of words to guess as a group (default: 4).
        :param previous_guesses: Set of previous guesses.
        :param metrics: Metrics object for tracking (optional).
        :param history: History of previous interactions (optional).
        :return: A tuple containing the guessed words and reasoning.
        """
        
        if metrics is None:
            metrics = Metrics()

        max_retries = 3  # Define a maximum number of retries to prevent infinite loops
        retries = 0

        while retries < max_retries:
            # Prepare the game board as a string
            game_board_str = ', '.join(word_bank)

            # Step 1: Guesser Agent generates a guess and category
            guesser_prompt = (
                f"Words: {game_board_str}\n"
                f"Find a group of {group_size} related words and provide a category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4\n"
                f"Category: category_name"
            )

            logger.info("GuesserAgent is generating a guess and category.")
            guesser_reply = self.guesser_agent.generate_reply(
                messages=[
                    {"role": "system", "content": self.guesser_agent.system_message},
                    {"role": "user", "content": guesser_prompt}
                ]
            )

            # Handle the response type from GuesserAgent
            guesser_reply_str = self._extract_reply_str(guesser_reply, "GuesserAgent")
            if not guesser_reply_str:
                logger.error("GuesserAgent failed to generate a valid reply.")
                raise ValueError("GuesserAgent failed to generate a valid reply.")

            # Parse GuesserAgent's reply
            try:
                guesser_group, guesser_category = self.parse_guesser_reply(guesser_reply_str)
                logger.info(f"GuesserAgent guessed group: {guesser_group} with category: {guesser_category}")
            except ValueError as e:
                logger.error(f"Error parsing GuesserAgent's reply: {e}")
                raise

            # Step 2: Validator Agent validates the category by finding a matching group
            validator_prompt = (
                f"Given the following words: {game_board_str}\n"
                f"And the category: {guesser_category}\n"
                f"Identify the 4 words that belong to this category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4"
            )

            logger.info("ValidatorAgent is validating the guess based on the category.")
            validator_reply = self.validator_agent.generate_reply(
                messages=[
                    {"role": "system", "content": self.validator_agent.system_message},
                    {"role": "user", "content": validator_prompt}
                ]
            )

            # Handle the response type from ValidatorAgent
            validator_reply_str = self._extract_reply_str(validator_reply, "ValidatorAgent")
            if not validator_reply_str:
                logger.error("ValidatorAgent failed to generate a valid reply.")
                raise ValueError("ValidatorAgent failed to generate a valid reply.")

            # Parse ValidatorAgent's reply
            try:
                validator_group = self.parse_validator_reply(validator_reply_str)
                logger.info(f"ValidatorAgent identified group: {validator_group}")
            except ValueError as e:
                logger.error(f"Error parsing ValidatorAgent's reply: {e}")
                raise

            # Step 3: Consensus Agent checks if both groups match
            consensus_prompt = (
                f"Guesser Group: {', '.join(guesser_group)}\n"
                f"Validator Group: {', '.join(validator_group)}\n"
                f"Determine if both groups contain exactly the same words, regardless of order.\n"
                f"Respond with 'Consensus reached' if they are identical, or 'Consensus not reached' otherwise."
            )

            logger.info("ConsensusAgent is checking if the groups match.")
            consensus_reply = self.consensus_agent.generate_reply(
                messages=[
                    {"role": "system", "content": self.consensus_agent.system_message},
                    {"role": "user", "content": consensus_prompt}
                ]
            )

            # Handle the response type from ConsensusAgent
            consensus_reply_str = self._extract_reply_str(consensus_reply, "ConsensusAgent")
            if not consensus_reply_str:
                logger.error("ConsensusAgent failed to generate a valid reply.")
                raise ValueError("ConsensusAgent failed to generate a valid reply.")

            # Parse ConsensusAgent's reply
            consensus_result = self.parse_consensus_reply(consensus_reply_str)
            logger.info(f"ConsensusAgent result: {consensus_result}")

            if consensus_result:
                # Consensus reached; submit the guess
                reasoning = guesser_category
                return tuple(guesser_group), reasoning
            else:
                # Consensus not reached; provide feedback to GuesserAgent
                feedback = "The Validator Agent identified a different group for the category. Please reconsider your guess."
                self.agent_memory['GuesserAgent']['feedback'] = feedback
                logger.info(f"Consensus not reached. Attempt {retries + 1} of {max_retries}.")

                # Increment retry counter
                retries += 1

                # Optionally, you can append to history or update metrics here
                history += f"Attempt {retries}: {feedback}\n"

        # If consensus is not reached after maximum retries
        logger.error("Consensus not reached after maximum retries. Unable to make a guess.")
        raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

    def _extract_reply_str(self, reply: Any, agent_name: str) -> Optional[str]:
        """
        Helper method to extract the reply string from the agent's response.

        :param reply: The raw reply from the agent (str or Dict).
        :param agent_name: Name of the agent (for logging purposes).
        :return: Extracted reply string if available, else None.
        """
        if isinstance(reply, str):
            return reply
        elif isinstance(reply, dict):
            extracted_reply = reply.get('reply')
            if isinstance(extracted_reply, str):
                return extracted_reply
            else:
                logger.warning(f"{agent_name} returned a dict without a 'reply' string.")
                return None
        else:
            logger.warning(f"{agent_name} returned an unsupported reply type: {type(reply)}.")
            return None

    def parse_guesser_reply(self, reply: str) -> Tuple[List[str], str]:
        """
        Parse the GuesserAgent's reply to extract the group and category.

        :param reply: The raw reply from GuesserAgent.
        :return: A tuple of the group of words and the category.
        """
        lines = reply.strip().split('\n')
        group_line = next((line for line in lines if line.startswith('Group:')), '')
        category_line = next((line for line in lines if line.startswith('Category:')), '')

        if not group_line or not category_line:
            raise ValueError("GuesserAgent's reply is missing 'Group' or 'Category'.")

        group = [word.strip() for word in group_line.replace('Group:', '').split(',')]
        category = category_line.replace('Category:', '').strip()

        if len(group) != 4:
            raise ValueError("GuesserAgent's group does not contain exactly 4 words.")

        return group, category

    def parse_validator_reply(self, reply: str) -> List[str]:
        """
        Parse the ValidatorAgent's reply to extract the validated group.

        :param reply: The raw reply from ValidatorAgent.
        :return: A list of words representing the validated group.
        """
        lines = reply.strip().split('\n')
        group_line = next((line for line in lines if line.startswith('Group:')), '')

        if not group_line:
            raise ValueError("ValidatorAgent's reply is missing 'Group'.")

        group = [word.strip() for word in group_line.replace('Group:', '').split(',')]

        if len(group) != 4:
            raise ValueError("ValidatorAgent's group does not contain exactly 4 words.")

        return group

    def parse_consensus_reply(self, reply: str) -> bool:
        """
        Parse the ConsensusAgent's reply to determine if consensus is reached.

        :param reply: The raw reply from ConsensusAgent.
        :return: True if consensus is reached, False otherwise.
        """
        normalized_reply = reply.strip().lower()
        if "consensus reached" in normalized_reply:
            return True
        elif "consensus not reached" in normalized_reply:
            return False
        else:
            logger.warning(f"Unexpected ConsensusAgent response: {reply}")
            return False  # Default to False or handle as needed

    def play(self, game: Connections, commit_to: Optional[str] = None) -> List[bool]:
        """
        Play the game using the GVCSolver.

        :param game: The Connections game instance.
        :param commit_to: Optional database to commit metrics.
        :return: List indicating which categories were solved.
        """
        self.game = game  # Make the game accessible to agents
        metrics = Metrics()
        previous_guesses: Set[Tuple[str, ...]] = set()
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
                logger.info(f"Guessed: {guess} --> {cat}")

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
                logger.warning(str(e))
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break

        if commit_to is not None:
            metrics.commit(to_db=commit_to)
        return game.solved_categories
