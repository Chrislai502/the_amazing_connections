import os
import logging
from typing import Optional, Any, List, Tuple, Dict, Set

from .solver import Solver
from ..game import Connections, GameOverException
from ..metrics import Metrics

from autogen import ConversableAgent
import pystache
import re

# Constants
MUSTACHE_FILENAMES = {
    "GuesserAgent": "prompts/gvc/guesser_agent.mustache",
    "ValidatorAgent": "prompts/gvc/validator_agent.mustache",
    "ConsensusAgent": "prompts/gvc/consensus_agent.mustache",
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RATING_SCALE = 5

class GVCSolver(Solver):
    def __init__(self):
        super().__init__()
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 0.7
            }]
        }
        # Initialize tracking dictionaries
        self.guesses: Dict[str, List[Tuple[str, ...]]] = {}  # category -> list of failed word groups
        
        # Guesser Past Understandings
        self.guesser_past_understandings = None
        
        # Initialize previous validator feedback
        self.prev_validator_feedback_if_rejected = None
        self.validator_dict = {}
        
        # Successful Guesses and failed guesses
        # self.successful_guesses = {}
        self.failed_guesses = {}
        
        # Cache Replies
        self.guesser_reply = None
        self.validator_reply = None
        self.remaining_str = None
        self.feedback = None
        
    # Import Agent System Prompts
    def get_prompts(self, group_size: int)-> Dict[str, str]:
        
        output_dict = {}
        
        # Load the .mustache file
        for key, value in MUSTACHE_FILENAMES.items():
            with open(value, "r") as f:
                template = f.read()

            # Variables to inject
            data = {
                "group_size": group_size,
                "rating_scale": RATING_SCALE,
            }

            # Render the template
            renderer = pystache.Renderer()
            output = renderer.render(template, data)
            output_dict[key] = output
        
        return output_dict

    def initialize_agents(self, system_messages):
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

        max_retries = 8
        
        self.initialize_agents(self.get_prompts(group_size))

        for attempt in range(1, max_retries + 1):
            # logger.info(f"SOLVER: Attempt {attempt} of {max_retries}")
            
            self.remaining_str = ', '.join(remaining_words)
            entire_str = ', '.join(entire_game_board)
            # successful = list(set(entire_game_board) - set(remaining_words))

            # Prepare feedback about unsuccessful and successful categories
            self.feedback = ""
            
            # Adding Unsuccessful Guesses
            if len(self.failed_guesses.keys()) > 0:
                self.feedback += (
                    "- The following categories represent word groups that have been guessed, but the Game Engine verified "
                    "that these specific groups are not part of the final solution. Note that the Game Engine evaluates only "
                    "the word groups, not the associated categories, which reflect the Guesser Agent's interpretation at the time of the guess. "
                    "The categories may or may not accurately describe the word groups:\n"
                )
                for category, word_groups in self.failed_guesses.items():
                    word_groups_str = '; '.join(word_groups)
                    self.feedback += f"  * {category}: {word_groups_str}\n"

            # # Adding Successful Guesses
            # if len(self.successful_guesses.keys()) > 0:
            #     self.feedback += f"- These are Successful Guesses! Don't repeat guesses for these words:\n"
            #     for category, word_groups in self.successful_guesses.items():
            #         word_groups_str = '; '.join(word_groups)
            #         self.feedback += f"  * {category}: {word_groups_str}\n"
            
            # Adding In previous understandings
            previous_understandings_str = ""
            if self.guesser_past_understandings is not None:
                previous_understandings_str += f"- This is your previous understanding of the board:\n"
                for category, word_groups in self.guesser_past_understandings.items():
                    word_groups_str = '; '.join(word_groups)
                    previous_understandings_str += f"  * {category}: {word_groups_str}\n"

            # Step 1: GuesserAgent generates a guess and category using remaining words
            
            # Building Guesser Agent Prompt
            guesser_prompt = (
                f"**Game Engine Feedback**\n"
                f"{self.feedback}\n"
            )
            
            if self.guesser_past_understandings:
                guesser_prompt += (
                    f"**Your Last Board Understanding**\n"
                    f"{previous_understandings_str}\n"
                )
                
            if self.prev_validator_feedback_if_rejected is not None:
                guesser_prompt += (
                    f"**Validator Feedback**\n"
                    f"Validator have rejected your guess. {self.prev_validator_feedback_if_rejected}\n\n"
                )
                
            guesser_prompt += (
                f"**Remaining Words**\n"
                f"Words: {self.remaining_str}\n"
            )
            
            logger.info(f"GUESSER PROMPT:\n\n{guesser_prompt}")
            logger.info("GuesserAgent: Generating a guess and category.")
            try:
                self.guesser_reply = self._get_agent_reply(self.guesser_agent, guesser_prompt, "GuesserAgent")
                # logger.info(f"GuesserAgent: Reply: {self.guesser_reply}")
                the_guess, self.guesser_past_understandings = self.parse_guesser_reply(self.guesser_reply)
                guesser_group, guesser_category = the_guess
                # logger.info(f"GuesserAgent: Understanding for this round: {self.guesser_past_understandings}")
                logger.info(f"GuesserAgent: Guessed group: {guesser_group} with category: {guesser_category}")
            except ValueError as e:
                logger.error(f"SOLVER: Error parsing GuesserAgent's reply: {e}")
                raise
            

            # Step 2: ValidatorAgent validates the category using the entire game board
            validator_prompt = (
                f"**Context:**\n"
                f"{self.guesser_reply}\n"
                f"**Remaining Words:**\n"
                f"Words: {self.remaining_str}\n"  
                "**Game Engine Feedback**\n"
                f"{self.feedback}\n"
            )


            logger.info("ValidatorAgent: Validating the guess based on the category.")
            try:
                self.validator_reply = self._get_agent_reply(self.validator_agent, validator_prompt, "ValidatorAgent")
                self.validator_dict = self.parse_validator_reply(self.validator_reply)
            except ValueError as e:
                logger.error(f"SOLVER: Error parsing ValidatorAgent's reply: {e}")
                raise

            if self.validator_dict["agreement"]:
                self.prev_validator_feedback_if_rejected = None
                # Consensus reached; record successful guess
                if guesser_category not in self.guesses:
                    self.guesses[guesser_category] = []
                self.guesses[guesser_category].append(tuple(guesser_group))
                logger.info(f"SOLVER: Consensus reached for category '{guesser_category}'.")
                logger.info(f"SOLVER: Attempt {attempt} of {max_retries}")
                return tuple(guesser_group), guesser_category
            else:
                self.prev_validator_feedback_if_rejected = self.validator_dict["validator_feedback"]
                logger.info(f"SOLVER: NO Consensus reached for category '{guesser_category}'. Attempt {attempt} of {max_retries}.")

        # If all retries are exhausted without consensus, Pick Validator's last Guess
        logger.info(f"SOLVER: Consensus not reached after {max_retries} retries, using ConsensusAgent's final guess.")
        consensus_prompt = (
            f"**Context:**\n"
            f"GuesserAgent:\n"
            f"{self.guesser_reply}\n"
            f"ValidatorAgent:\n"
            f"{self.validator_reply}\n"
            f"**Remaining Words:**\n"
            f"Words: {self.remaining_str}\n"  
            "**Game Engine Feedback**\n"
            f"{self.feedback}\n"
        )
        try:
            consensus_reply = self._get_agent_reply(self.consensus_agent, consensus_prompt, "ConsensusAgent")
            consensus_group, consensus_category = self.parse_consensus_reply(consensus_reply)
        except ValueError as e:
            logger.error(f"SOLVER: Error parsing ConsensusAgent's reply: {e}")
            raise
        logger.info(f"ConsensusAgent: Guessed group: {consensus_group} with category: {consensus_category}")
        return tuple(consensus_group), consensus_category
        
        # raise ValueError("Consensus not reached after maximum retries. Unable to make a guess.")

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
        # logger.debug(f"{agent_name} raw reply: {reply}")  # Log raw reply for debugging
        # logger.info(f"{agent_name} raw reply: {reply}")
        reply_str = self._extract_reply_str(reply, agent_name)
        if not reply_str:
            logger.error(f"{agent_name} failed to generate a valid reply.")
            raise ValueError(f"{agent_name} failed to generate a valid reply.")
        # logger.info(f"{agent_name} reply_str: {reply_str}")
        
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
                return extracted_reply
            else:
                logger.warning(f"{agent_name} returned a dict without a 'reply' string.")
                return None
        else:
            logger.warning(f"{agent_name} returned an unsupported reply type: {type(reply)}.")
            return None

    def parse_guesser_reply(self, reply: str) -> Tuple[Tuple[List[str], str], Dict[str, List[str]]]:
        """
        Parse the GuesserAgent's reply to extract the guessed group, category, and overall board understanding.

        :param reply: The raw reply from the GuesserAgent.
        :return: A tuple containing:
                    1. A tuple with the guesser's final guessed group (list of words) and its category (string).
                    2. A dictionary where keys are category descriptions and values are lists of words for each understanding group.
        :raises ValueError: If the reply format is incorrect or required sections are missing.
        """
        try:
            # Normalize the input to remove extra spaces, blank lines, or unexpected text
            normalized_reply = re.sub(r"\s*\n\s*", "\n", reply.strip())  # Normalize spaces and newlines
            normalized_reply = re.sub(r"Below are the guesses:.*", "", normalized_reply, flags=re.DOTALL)  # Remove extra text between sections

            # Extract <UNDERSTANDING_OF_BOARD> section
            understanding_section = re.search(
                r"<UNDERSTANDING_OF_BOARD>(.*?)<GUESS_FOR_THIS_ROUND>",
                normalized_reply,
                re.DOTALL
            )
            if not understanding_section:
                raise ValueError("Missing <UNDERSTANDING_OF_BOARD> section.")
            understandings_text = understanding_section.group(1).strip()

            # Regex pattern to parse each group and its category description
            understanding_pattern = re.compile(
                r"Group\d+: (.*?)\nCategory Description: (.*?)(?:\n|$)",
                re.DOTALL
            )

            # Parse groups and categories into a dictionary
            understandings = {
                match.group(2).strip(): [word.strip() for word in re.split(r",\s*", match.group(1))]
                for match in understanding_pattern.finditer(understandings_text)
            }

            # Extract <GUESS_FOR_THIS_ROUND> section
            guess_section = re.search(r"<GUESS_FOR_THIS_ROUND>(.*)", normalized_reply, re.DOTALL)
            if not guess_section:
                raise ValueError("Missing <GUESS_FOR_THIS_ROUND> section.")
            guess_text = guess_section.group(1).strip()

            # Parse the final guess group and its category
            final_guess_pattern = re.compile(r"Group: (.*?)\nCategory: (.*)")
            final_guess_match = final_guess_pattern.search(guess_text)
            if not final_guess_match:
                raise ValueError("Final guess format is incorrect.")

            final_group = [word.strip() for word in re.split(r",\s*", final_guess_match.group(1))]
            final_category = final_guess_match.group(2).strip()

            return (final_group, final_category), understandings

        except Exception as e:
            raise ValueError(f"Error parsing reply: {str(e)}")

    def parse_validator_reply(self, reply: str) -> Dict[str, Any]:
        """
        Parse the validator agent's response to extract the validation report.

        :param reply: The raw reply from the validator agent.
        :return: A dictionary with the validation report, containing:
                - "agreement" (bool): Whether the validator agrees with the guess.
                - "correctness_rating" (int): Rating of the correctness of the guesser's interpretation.
                - "confidence_rating" (int): Rating of the guesser's confidence.
        :raises ValueError: If the reply format is incorrect or missing required fields.
        """
        try:
            # Extract "Agreement to Perform the Guess" (True/False)
            agreement_match = re.search(r"Agreement to Perform the Guess:\s*(True|False)", reply)
            if not agreement_match:
                raise ValueError("Missing 'Agreement to Perform the Guess' field.")
            agreement = agreement_match.group(1) == "True"

            # Extract "Rating of correctness"
            correctness_match = re.search(
                r"Rating of guesser agent's correctness of interpreting the board out of \d+:\s*(\d+)", reply
            )
            if not correctness_match:
                raise ValueError("Missing 'Rating of correctness' field.")
            correctness_rating = int(correctness_match.group(1))

            # Extract "Rating of confidence"
            confidence_match = re.search(
                r"Rating of guesser agent's confidence in the guesses out of \d+:\s*(\d+)", reply
            )
            if not confidence_match:
                raise ValueError("Missing 'Rating of confidence' field.")
            confidence_rating = int(confidence_match.group(1))

            # Extract "Feedback for Guesser Agent"
            feedback_match = re.search(
                r"Feedback for Guesser Agent:\s*(.*?)(?:\n<|$)", reply, re.DOTALL
            )
            if not feedback_match:
                # raise ValueError("Missing 'Feedback for Guesser Agent' field.")
                validator_feedback = ""
            else:
                validator_feedback = feedback_match.group(1)

            # Return extracted data
            return {
                "agreement": agreement,
                "correctness_rating": correctness_rating,
                "confidence_rating": confidence_rating,
                "validator_feedback": validator_feedback
            }

        except Exception as e:
            raise ValueError(f"Error parsing validation report: {str(e)}")

    def parse_consensus_reply(self, reply: str) -> Tuple[List[str], str]:
        try:
            # Extract final guess from <GUESS_FOR_THIS_ROUND> section
            guess_section = re.search(r"<GUESS_FOR_THIS_ROUND>(.*)", reply, re.DOTALL)
            if not guess_section:
                raise ValueError("Missing <GUESS_FOR_THIS_ROUND> section.")
            guess_text = guess_section.group(1).strip()

            # Parse the final guess group and its category
            final_guess_pattern = re.compile(r"Group: (.*?)\nCategory: (.*)")
            final_guess_match = final_guess_pattern.search(guess_text)
            if not final_guess_match:
                raise ValueError("Final guess format is incorrect.")
            
            final_group = [word.strip() for word in final_guess_match.group(1).split(",")]
            final_category = final_guess_match.group(2).strip()

            return final_group, final_category

        except Exception as e:
            raise ValueError(f"Error parsing reply: {str(e)}")

    def play(self, game: Connections, commit_to: Optional[str] = None) -> List[bool]:
        """
        Play the game using the GVCSolver.

        :param game: The Connections game instance.
        :param commit_to: Optional database to commit metrics.
        :return: List indicating which categories were solved.
        """
        metrics = Metrics()
        previous_guesses: Set[Tuple[str, ...]] = set()
        entire_game_board = list(game.all_words)  # Capture the entire game board at start
        try_counter = 0
        while not game.is_over:
            try:
                remaining_words = game.all_words  # Current remaining words
                guess, reasoning = self.guess(
                    remaining_words=remaining_words,
                    entire_game_board=entire_game_board,
                    group_size=game.group_size,
                    metrics=metrics
                )
                # Attempt to check the guess
                cat = game.category_guess_check(list(guess))
                logger.info(f"GAME ENGINE: Guessed: {guess} --> {cat}")
    
                if cat is None: # If the guess is wrong
                    previous_guesses.add(tuple(guess))
                    metrics.hallucination_words(list(guess), remaining_words)
                    metrics.increment_failed_guesses()
                    self.failed_guesses[reasoning] = guess
                else: # If the guess is correct
                    guessed_cat_idx = game._og_groups.index(cat)
                    metrics.add_solve(level=guessed_cat_idx)
                    metrics.cosine_similarity_category(guessed_cat=reasoning, correct_cat=cat.group)
                    # self.successful_guesses[cat.group] = cat.members
                    # No need to modify 'game.all_words' manually
                try_counter += 1
                if try_counter == 6:
                    raise NotImplementedError
            except GameOverException as e:
                logger.warning(str(e))
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break
            
            # "Reset the State of the agents"
            self.guesser_past_understandings = None
            self.prev_validator_feedback_if_rejected = None
            self.validator_dict = {}
            self.guesser_reply = None
            self.remaining_str = None
            self.feedback = None

        if commit_to:
            metrics.commit(to_db=commit_to)
        return game.solved_categories
