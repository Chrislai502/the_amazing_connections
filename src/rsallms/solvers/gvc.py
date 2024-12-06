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
        self.guesses: Dict[str, List[Tuple[str, ...]]] = {}
        self.initialize_agents()

    def initialize_agents(self):
        system_messages = {
    "GuesserAgent": (
        "You are an Expert Word Grouping Agent. You understand literature, culture, and are well-versed in common phrases and wordplay. Given a list of words, "
        "propose a group of 4 related words and a corresponding category. Try your best to select a category that only encompasses four words. Refer to the following examples of categories and word groups as guidance: "
        "\n\nSNEAKER BRANDS: adidas, nike, puma, reebok"
        "\nMUSICALS BEGINNING WITH “C”: cabaret, carousel, cats, chicago"
        "\nCLEANING VERBS: dust, mop, sweep, vacuum"
        "\n___ MAN SUPERHEROES: bat, iron, spider, super"
        "\n\nSTREAMING SERVICES: hulu, netflix, peacock, prime"
        "\nCONDIMENTS: ketchup, mayo, relish, tartar"
        "\nSYNONYMS FOR SAD: blue, down, glum, low"
        "\nCLUE CHARACTERS: green, mustard, plum, scarlet"
        "\n\nMONOPOLY SQUARES: boardwalk, chance, go, jail"
        "\nSHADES OF BLUE: baby, midnight, powder, royal"
        "\nRAPPERS: common, future, ice cube, q-tip"
        "\nMEMBERS OF A SEPTET: sea, sin, sister, wonder"
        "\n\nLEG PARTS: ankle, knee, shin, thigh"
        "\nBABY ANIMALS: calf, cub, joey, kid"
        "\nSLANG FOR TOILET: can, head, john, throne"
        "\n___ FISH THAT AREN’T FISH: cray, jelly, silver, star"
    ),
    "ValidatorAgent": (
        "You are an Expert Word Grouping Agent. You understand literature, culture, and are well-versed in common phrases and wordplay. Given a list of words, "
        "and a category, find the **4 words** that best fit the category. Refer to the following examples of categories and word groups as guidance: "
        "\n\nSNEAKER BRANDS: adidas, nike, puma, reebok"
        "\nMUSICALS BEGINNING WITH “C”: cabaret, carousel, cats, chicago"
        "\nCLEANING VERBS: dust, mop, sweep, vacuum"
        "\n___ MAN SUPERHEROES: bat, iron, spider, super"
        "\n\nSTREAMING SERVICES: hulu, netflix, peacock, prime"
        "\nCONDIMENTS: ketchup, mayo, relish, tartar"
        "\nSYNONYMS FOR SAD: blue, down, glum, low"
        "\nCLUE CHARACTERS: green, mustard, plum, scarlet"
        "\n\nMONOPOLY SQUARES: boardwalk, chance, go, jail"
        "\nSHADES OF BLUE: baby, midnight, powder, royal"
        "\nRAPPERS: common, future, ice cube, q-tip"
        "\nMEMBERS OF A SEPTET: sea, sin, sister, wonder"
        "\n\nLEG PARTS: ankle, knee, shin, thigh"
        "\nBABY ANIMALS: calf, cub, joey, kid"
        "\nSLANG FOR TOILET: can, head, john, throne"
        "\n___ FISH THAT AREN’T FISH: cray, jelly, silver, star"
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
        """Reset the GVCSolver's tracking state for a new game."""
        self.guesses.clear()
        logger.info("GVCSolver has been reset. Tracking sets cleared.")

    def guess(
        self, 
        remaining_words: List[str], 
        entire_game_board: List[str],
        group_size: int = 4, 
        metrics: Optional[Metrics] = None
    ) -> Tuple[Tuple[str, ...], str]:
        """
        Make a guess using the Guesser, Validator, and Consensus agents.

        :param remaining_words: Current list of remaining words in the game.
        :param entire_game_board: The complete list of words in the game.
        :param group_size: Number of words to guess as a group (default: 4).
        :param metrics: Metrics object for tracking (optional).
        :return: A tuple containing the guessed words and the category.
        :raises ValueError: If consensus is not reached after maximum retries.
        """
        metrics = metrics or Metrics()
        max_retries = 15

        for attempt in range(1, max_retries + 1):
            feedback = self._generate_feedback(entire_game_board, remaining_words)

            # Step 1: GuesserAgent generates a guess and category
            guesser_prompt = self._create_guesser_prompt(remaining_words, group_size, feedback)
            logger.info("GuesserAgent is generating a guess and category.")
            try:
                guesser_reply = self._get_agent_reply(self.guesser_agent, guesser_prompt, "GuesserAgent")
                guesser_group, guesser_category = self.parse_guesser_reply(guesser_reply)
                logger.info(f"GuesserAgent guessed group: {guesser_group} with category: {guesser_category}")
            except ValueError as e:
                logger.error(f"Error parsing GuesserAgent's reply: {e}")
                raise

            # Step 2: ValidatorAgent validates the category using the entire game board
            validator_prompt = self._create_validator_prompt(remaining_words, guesser_category, feedback)
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
                "Determine if both groups contain exactly the same words, regardless of order.\n"
                "Respond with 'Consensus reached' if they are identical, or 'Consensus not reached' otherwise."
            )
            logger.info("ConsensusAgent is checking if the groups match.")
            consensus_reply = self._get_agent_reply(self.consensus_agent, consensus_prompt, "ConsensusAgent")
            consensus_result = self.parse_consensus_reply(consensus_reply)
            logger.info(f"ConsensusAgent result: {consensus_result}")

            if consensus_result:
                self.guesses.setdefault(guesser_category, []).append(tuple(guesser_group))
                logger.info(f"Consensus reached for category '{guesser_category}'.")
                return tuple(guesser_group), guesser_category
            else:
                logger.info(f"Consensus not reached for category '{guesser_category}'. Attempt {attempt} of {max_retries}.")
                self.guesses.setdefault(guesser_category, []).append(tuple(guesser_group))
                if attempt == max_retries:
                    logger.error("Consensus not reached after maximum retries. Unable to make a guess.")
                    raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

    def _generate_feedback(self, entire_game_board: List[str], remaining_words: List[str]) -> str:
        """Generate feedback based on previous guesses."""
        feedback = ""
        if self.guesses:
            feedback += "Note:\n"
            feedback += "- Be aware that the following category and word group pairs either do not match or the category isn't specific enough. **Do not repeat categories**!:\n"
            for category, word_groups in self.guesses.items():
                word_groups_str = '; '.join(['(' + ', '.join(group) + ')' for group in word_groups])
                feedback += f"  * {category}: {word_groups_str}\n"
        return feedback

    def _create_guesser_prompt(self, remaining_words: List[str], group_size: int, feedback: str) -> str:
        """Create the prompt for the GuesserAgent."""
        remaining_str = ', '.join(remaining_words)
        return (
            f"{feedback}\n"
            f"Words: {remaining_str}\n\n"
            f"**Objective:**\n"
            f"Find a group of {group_size} related words from the list above and provide a specific category that unambiguously describes the relationship between these words.\n\n"
            f"**Guidelines:**\n"
            f"- Categories must be more specific than broad classifications like 'Names', 'Verbs', or '5-Letter Words'.\n"
            f"- Each word in the group should clearly fit the category without overlapping into multiple categories. Be aware of common phrases, word play, and words with multiple meanings.\n"
            f"- Avoid categories that are too vague or general and might encompass other words.\n\n"
            f"**Examples:**\n"
            f"1. **Group:** Bass, Flounder, Salmon, Trout\n"
            f"   **Category:** Fish\n\n"
            f"2. **Group:** Ant, Drill, Island, Opal\n"
            f"   **Category:** Fire ___\n\n"
            f"**Format Your Response As Follows:**\n"
            f"```\n"
            f"Group: word1, word2, word3, word4\n"
            f"Category: category_name\n"
            f"```\n"
            f"Ensure there is no additional text or explanation beyond the specified format."
        )

    def _create_validator_prompt(self, remaining_words: List[str], category: str, feedback: str) -> str:
        """Create the prompt for the ValidatorAgent."""
        remaining_str = ', '.join(remaining_words)
        return (
            f"{feedback}\n"
            f"**Words:** {remaining_str}\n"
            f"**Category:** {category}\n\n"
            f"**Objective:**\n"
            f"Identify the four words from the list above that belong to the specified category. Ensure that each selected word clearly fits the category without ambiguity.\n\n"
            f"**Guidelines:**\n"
            f"- The category is already provided and is specific. Focus solely on selecting the words that best match this category. Be aware of word play and words with multiple meanings.\n"
            f"- Each word should unambiguously fit the category. Avoid selecting words that could belong to multiple categories unless they are a perfect fit.\n"
            f"- Ensure that exactly four words are selected. Do not include more or fewer words.\n"
            f"- Avoid using additional explanations or commentary. Only provide the required output in the specified format.\n\n"
            f"**Examples:**\n"
            f"1. **Category:** Fish\n"
            f"   **Words:** Bass, Flounder, Salmon, Trout, Ant, Drill, Island, Opal\n"
            f"   **Group:** Bass, Flounder, Salmon, Trout\n\n"
            f"2. **Category:** Fire ___\n"
            f"   **Words:** Ant, Drill, Island, Opal, Fire, Water, Earth, Air\n"
            f"   **Group:** Ant, Drill, Island, Opal\n\n"
            f"**Format Your Response As Follows:**\n"
            f"```\n"
            f"Group: word1, word2, word3, word4\n"
            f"```\n"
            f"Ensure there is no additional text or explanation beyond the specified format."
        )

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
        logger.debug(f"{agent_name} raw reply: {reply}")
        reply_str = self._extract_reply_str(reply, agent_name)
        if not reply_str:
            logger.error(f"{agent_name} failed to generate a valid reply.")
            raise ValueError(f"{agent_name} failed to generate a valid reply.")
        return reply_str

    def _extract_reply_str(self, reply: Any, agent_name: str) -> Optional[str]:
        """
        Extracts the reply string from the agent's response.

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
            logger.warning(f"{agent_name} returned a dict without a 'reply' string.")
        else:
            logger.warning(f"{agent_name} returned an unsupported reply type: {type(reply)}.")
        return None

    def parse_guesser_reply(self, reply: str) -> Tuple[List[str], str]:
        """
        Parse the GuesserAgent's reply to extract the group and category.

        :param reply: The raw reply from GuesserAgent.
        :return: A tuple of the group of words and the category.
        :raises ValueError: If the reply format is incorrect.
        """
        lines = reply.strip().split('\n')
        group_line = next((line for line in lines if line.startswith('Group:')), '')
        category_line = next((line for line in lines if line.startswith('Category:')), '')

        if not group_line or not category_line:
            missing = 'Group' if not group_line else 'Category'
            raise ValueError(f"GuesserAgent's reply is missing '{missing}'.")

        group = [word.strip() for word in group_line.replace('Group:', '').split(',')]
        category = category_line.replace('Category:', '').strip()

        if len(group) != 4:
            raise ValueError(f"GuesserAgent's group contains {len(group)} words; expected exactly 4.")

        return group, category

    def parse_validator_reply(self, reply: str) -> List[str]:
        """
        Parse the ValidatorAgent's reply to extract the validated group.

        :param reply: The raw reply from ValidatorAgent.
        :return: A list of words representing the validated group.
        :raises ValueError: If the reply format is incorrect.
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
        normalized = reply.strip().lower()
        if "consensus reached" in normalized:
            return True
        elif "consensus not reached" in normalized:
            return False
        logger.warning(f"Unexpected ConsensusAgent response: '{reply}'. Assuming consensus not reached.")
        return False

    def play(self, game: Connections, commit_to: Optional[str] = None) -> List[bool]:
        """
        Play the game using the GVCSolver.

        :param game: The Connections game instance.
        :param commit_to: Optional database to commit metrics.
        :return: List indicating which categories were solved.
        """
        metrics = Metrics()
        entire_game_board = list(game.all_words)

        while not game.is_over:
            try:
                remaining_words = game.all_words
                guess, category = self.guess(
                    remaining_words=remaining_words,
                    entire_game_board=entire_game_board,
                    group_size=game.group_size,
                    metrics=metrics
                )
                cat = game.category_guess_check(list(guess))
                logger.info(f"Guessed: {guess} --> {cat}")

                if cat is None:
                    metrics.hallucination_words(list(guess), remaining_words)
                    metrics.increment_failed_guesses()
                else:
                    guessed_cat_idx = game._og_groups.index(cat)
                    metrics.add_solve(level=guessed_cat_idx)
                    metrics.cosine_similarity_category(guessed_cat=category, correct_cat=cat.group)
            except GameOverException as e:
                logger.warning(str(e))
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break

        if commit_to:
            metrics.commit(to_db=commit_to)
        return game.solved_categories
