import os
import logging
from typing import Optional, Any, List, Tuple, Dict, Set

from .solver import Solver
from ..game import Connections, GameOverException
from ..metrics import Metrics

from autogen import ConversableAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GVCSolver(Solver):
    def __init__(self):
        super().__init__()
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY")
            }]
        }
        # Initialize tracking dictionaries
        self.unsuccessful_guesses: Dict[str, List[Tuple[str, ...]]] = {}  # category -> list of failed word groups
        self.successful_guesses: Dict[str, Tuple[str, ...]] = {}        # category -> successful word group
        self.initialize_agents()

    def initialize_agents(self):
        system_messages = {
            "GuesserAgent": (
                "You are a Guesser Agent in a word game. Given a list of words, "
                "propose a group of 4 related words and a corresponding category."
            ),
            "ValidatorAgent": (
                "You are a Validator Agent. Given a list of words and a category, "
                "identify a group of 4 words that fit the category."
            ),
            "ConsensusAgent": (
                "You are a Consensus Agent. Compare two groups of words and determine if they contain exactly the same words, regardless of their order. Respond with 'Consensus reached' if both groups have identical words, or 'Consensus not reached' otherwise."
            )
        }

        self.guesser_agent = ConversableAgent(
            name="GuesserAgent",
            system_message=system_messages["GuesserAgent"],
            llm_config=self.llm_config,
        )
        self.validator_agent = ConversableAgent(
            name="ValidatorAgent",
            system_message=system_messages["ValidatorAgent"],
            llm_config=self.llm_config,
        )
        self.consensus_agent = ConversableAgent(
            name="ConsensusAgent",
            system_message=system_messages["ConsensusAgent"],
            llm_config=self.llm_config,
        )

    def reset(self):
        """
        Reset the GVCSolver's tracking state for a new game.
        """
        self.unsuccessful_guesses.clear()
        self.successful_guesses.clear()
        logger.info("GVCSolver has been reset. Tracking sets cleared.")

    def guess(
        self, 
        remaining_words: List[str], 
        entire_game_board: List[str],
        group_size: int = 4, 
        metrics: Optional[Metrics] = None
    ) -> Tuple[Tuple[str, ...], str]:
        """
        Make a guess using the Guesser, Validator, and Consensus agents, ensuring that
        previously unsuccessful and successful categories are not repeated.

        :param remaining_words: Current list of remaining words in the game.
        :param entire_game_board: The complete list of words in the game.
        :param group_size: Number of words to guess as a group (default: 4).
        :param metrics: Metrics object for tracking (optional).
        :return: A tuple containing the guessed words and the category.
        :raises ValueError: If consensus is not reached after maximum retries.
        """
        if metrics is None:
            metrics = Metrics()

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            remaining_str = ', '.join(remaining_words)
            entire_str = ', '.join(entire_game_board)

            # Prepare feedback about unsuccessful and successful categories
            feedback = ""
            if self.unsuccessful_guesses or self.successful_guesses:
                feedback += "Note:\n"
                if self.unsuccessful_guesses:
                    feedback += "- Be aware that the following category and word group pairs either do not match or the category isn't specific enough:\n"
                    for category, word_groups in self.unsuccessful_guesses.items():
                        word_groups_str = '; '.join(['(' + ', '.join(group) + ')' for group in word_groups])
                        feedback += f"  * {category}: {word_groups_str}\n"
                if self.successful_guesses:
                    feedback += "- Do not repeat the following successfully guessed categories and ensure your guessed category doesn't overlap with any of these words:\n"
                    for category, word_groups in self.successful_guesses.items():
                        # word_groups_str = '; '.join(['(' + ', '.join(group) + ')' for group in word_groups])
                        feedback += f"  * {category}: {word_groups}\n"
                print(feedback)

            # Step 1: GuesserAgent generates a guess and category using remaining words
            guesser_prompt = (
                f"{feedback}"
                f"Words: {remaining_str}\n"
                f"Find a group of {group_size} related words and provide a category. Ensure the category is as specific as possible to the group of words, so there is no confusion of which words belong to said category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4\n"
                f"Category: category_name"
            )

            logger.info("GuesserAgent is generating a guess and category.")
            try:
                guesser_reply = self._get_agent_reply(self.guesser_agent, guesser_prompt, "GuesserAgent")
                guesser_group, guesser_category = self.parse_guesser_reply(guesser_reply)
                logger.info(f"GuesserAgent guessed group: {guesser_group} with category: {guesser_category}")
            except ValueError as e:
                logger.error(f"Error parsing GuesserAgent's reply: {e}")
                raise

            # Step 2: ValidatorAgent validates the category using the entire game board
            validator_prompt = (
                f"Given the following words: {entire_str}\n"
                f"And the category: {guesser_category}\n"
                f"Identify the 4 words that belong to this category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4"
            )

            logger.info("ValidatorAgent is validating the guess based on the category.")
            try:
                validator_reply = self._get_agent_reply(self.validator_agent, validator_prompt, "ValidatorAgent")
                validator_group = self.parse_validator_reply(validator_reply)
                logger.info(f"ValidatorAgent identified group: {validator_group}")
            except ValueError as e:
                logger.error(f"Error parsing ValidatorAgent's reply: {e}")
                raise

            # Step 3: ConsensusAgent checks if both groups match
            consensus_prompt = (
                f"Guesser Group: {', '.join(guesser_group)}\n"
                f"Validator Group: {', '.join(validator_group)}\n"
                f"Determine if both groups contain exactly the same words, regardless of order.\n"
                f"Respond with 'Consensus reached' if they are identical, or 'Consensus not reached' otherwise."
            )

            logger.info("ConsensusAgent is checking if the groups match.")
            consensus_reply = self._get_agent_reply(self.consensus_agent, consensus_prompt, "ConsensusAgent")
            consensus_result = self.parse_consensus_reply(consensus_reply)
            logger.info(f"ConsensusAgent result: {consensus_result}")

            if consensus_result:
                # Consensus reached; record successful guess
                self.successful_guesses[guesser_category] = tuple(guesser_group)
                logger.info(f"Consensus reached for category '{guesser_category}'.")
                return tuple(guesser_group), guesser_category
            else:
                # Consensus not reached; record unsuccessful guess
                logger.info(f"Consensus not reached for category '{guesser_category}'. Attempt {attempt} of {max_retries}.")
                if guesser_category not in self.unsuccessful_guesses:
                    self.unsuccessful_guesses[guesser_category] = []
                self.unsuccessful_guesses[guesser_category].append(tuple(guesser_group))
                if attempt == max_retries:
                    logger.error("Consensus not reached after maximum retries. Unable to make a guess.")
                    raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")
                # Optionally, continue to the next attempt without adding additional feedback

        # If all retries are exhausted without consensus
        raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

    def _get_agent_reply(self, agent: ConversableAgent, prompt: str, agent_name: str) -> str:
        """
        Sends a prompt to an agent and retrieves the response as a string.

        :param agent: The agent to interact with.
        :param prompt: The user prompt to send to the agent.
        :param agent_name: Name of the agent (for logging purposes).
        :return: The agent's reply as a string.
        :raises ValueError: If the agent fails to generate a valid reply.
        """
        reply = agent.generate_reply(
            messages=[
                {"role": "system", "content": agent.system_message},
                {"role": "user", "content": prompt}
            ]
        )
        logger.debug(f"{agent_name} raw reply: {reply}")  # Log raw reply for debugging
        reply_str = self._extract_reply_str(reply, agent_name)
        if not reply_str:
            logger.error(f"{agent_name} failed to generate a valid reply.")
            raise ValueError(f"{agent_name} failed to generate a valid reply.")
        return reply_str

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
             