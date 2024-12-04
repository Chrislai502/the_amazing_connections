import os
import logging
from typing import Optional, Any, List, Tuple, Set

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

    def guess(
        self, 
        remaining_words: List[str], 
        entire_game_board: List[str],
        group_size: int = 4, 
        metrics: Optional[Metrics] = None
    ) -> Tuple[Tuple[str, ...], str]:
        if metrics is None:
            metrics = Metrics()

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            remaining_str = ', '.join(remaining_words)
            entire_str = ', '.join(entire_game_board)

            # Step 1: GuesserAgent generates a guess and category using remaining words
            guesser_prompt = (
                f"Words: {remaining_str}\n"
                f"Find a group of {group_size} related words and provide a category. Ensure the category is as specific as possible to the group of words, so there is no confusion of which words belong to said category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4\n"
                f"Category: category_name"
            )

            logger.info("GuesserAgent is generating a guess and category.")
            guesser_reply = self._get_agent_reply(self.guesser_agent, guesser_prompt, "GuesserAgent")
            guesser_group, guesser_category = self.parse_guesser_reply(guesser_reply)
            logger.info(f"GuesserAgent guessed group: {guesser_group} with category: {guesser_category}")

            # Step 2: ValidatorAgent validates the category using the entire game board
            validator_prompt = (
                f"Given the following words: {entire_str}\n"
                f"And the category: {guesser_category}\n"
                f"Identify the 4 words that belong to this category.\n"
                f"Format:\n"
                f"Group: word1, word2, word3, word4"
            )

            logger.info("ValidatorAgent is validating the guess based on the category.")
            validator_reply = self._get_agent_reply(self.validator_agent, validator_prompt, "ValidatorAgent")
            validator_group = self.parse_validator_reply(validator_reply)
            logger.info(f"ValidatorAgent identified group: {validator_group}")

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
                return tuple(guesser_group), guesser_category
            else:
                logger.info(f"Consensus not reached. Attempt {attempt} of {max_retries}.")
                if attempt == max_retries:
                    logger.error("Consensus not reached after maximum retries. Unable to make a guess.")
                    raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

        raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

    def _get_agent_reply(self, agent: ConversableAgent, prompt: str, agent_name: str) -> str:
        reply = agent.generate_reply(
            messages=[
                {"role": "system", "content": agent.system_message},
                {"role": "user", "content": prompt}
            ]
        )
        reply_str = self._extract_reply_str(reply, agent_name)
        if not reply_str:
            logger.error(f"{agent_name} failed to generate a valid reply.")
            raise ValueError(f"{agent_name} failed to generate a valid reply.")
        return reply_str

    def _extract_reply_str(self, reply: Any, agent_name: str) -> Optional[str]:
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
        lines = reply.strip().split('\n')
        group_line = next((line for line in lines if line.startswith('Group:')), '')
        if not group_line:
            raise ValueError("ValidatorAgent's reply is missing 'Group'.")
        group = [word.strip() for word in group_line.replace('Group:', '').split(',')]
        if len(group) != 4:
            raise ValueError("ValidatorAgent's group does not contain exactly 4 words.")
        return group

    def parse_consensus_reply(self, reply: str) -> bool:
        normalized = reply.strip().lower()
        if "consensus reached" in normalized:
            return True
        elif "consensus not reached" in normalized:
            return False
        logger.warning(f"Unexpected ConsensusAgent response: {reply}")
        return False

    def play(self, game: Connections, commit_to: Optional[str] = None) -> List[bool]:
        metrics = Metrics()
        previous_guesses: Set[Tuple[str, ...]] = set()
        entire_game_board = list(game.all_words)  # Assuming game.all_words contains all words initially

        while not game.is_over:
            try:
                remaining_words = game.all_words.copy()  # Remaining words
                guess, reasoning = self.guess(
                    remaining_words=remaining_words,
                    entire_game_board=entire_game_board,
                    group_size=game.group_size,
                    metrics=metrics
                )
                cat = game.category_guess_check(list(guess))
                logger.info(f"Guessed: {guess} --> {cat}")

                if cat is None:
                    previous_guesses.add(tuple(guess))
                    metrics.hallucination_words(list(guess), game.all_words)
                    metrics.increment_failed_guesses()
                else:
                    guessed_cat_idx = game._og_groups.index(cat)
                    metrics.add_solve(level=guessed_cat_idx)
                    metrics.cosine_similarity_category(guessed_cat=reasoning, correct_cat=cat.group)
                    # Remove guessed words from the remaining words
                    game.all_words = [word for word in game.all_words if word not in guess]
            except GameOverException as e:
                logger.warning(str(e))
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break

        if commit_to:
            metrics.commit(to_db=commit_to)
        return game.solved_categories
