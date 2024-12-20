import os
import logging
from typing import Optional, Any, List, Tuple, Dict, Set

from .solver import Solver
from ..game import Connections, GameOverException
from ..metrics import Metrics

from autogen import ConversableAgent
import pystache
import re

from collections import deque  # For implementing the ring buffer

# Constants
MUSTACHE_FILENAMES = {
    "GuesserAgent": "prompts/gvc/guesser_agent.mustache",
    "ValidatorAgent": "prompts/gvc/validator_agent.mustache",
    # "ConsensusAgent": "prompts/gvc/consensus_agent.mustache",
    # "GroundingAgent": "prompts/gvc/grounding_agent.mustache",
    "SnapGuesserAgent": "prompts/gvc/snap_agent.mustache"
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RATING_SCALE = 5

class SGVCSolver(Solver):
    def __init__(self, api_type: str = "oai", model="gpt-4o"):
        super().__init__()
        
        if model== "gpt-4o-mini":
            self.conservative_llm_config = {
                "config_list": [{
                    "model": "gpt-4o-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "temperature": 1.00
                }]
            }
        else:
            self.conservative_llm_config = {
                "config_list": [{
                    "model": "gpt-4o",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "temperature": 1.00
                }]
            }
        
        self.snap_llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 1.0
            }]
        }
        
        # Initialize tracking dictionaries
        self.guesses: Dict[str, List[Tuple[str, ...]]] = {}  # category -> list of failed word groups
        
        # Guesser Past Understandings
        self.guesser_past_understandings = None
        self.last_guess = None
        
        # Initialize ring buffer for rejected guesses
        self.max_retries = 15
        self.rejected_guesses_buffer = deque(maxlen=self.max_retries)  # Ring buffer for last 5 rejected guesses
        
        # Initialize previous validator feedback
        self.prev_validator_feedback_if_rejected = None
        self.validator_dict = {}
        
        # Successful Guesses and failed guesses
        # self.successful_guesses = {}
        self.failed_guesses = {}
        self.sorted_failed_guesses = []
        
        # Cache Replies
        self.guesser_reply = None
        self.validator_reply = None
        self.remaining_str = None
        self.feedback = None
        
        # Max rounds per for Conservative Guessing Phase
        self.max_conservative_round_errors = 2
        self.max_conservative_wrong_guesses = 3
        self.snap_correct = False
        
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
            llm_config=self.conservative_llm_config,
            human_input_mode= "NEVER"
        )
        self.validator_agent = ConversableAgent(
            name="ValidatorAgent",
            system_message=system_messages["ValidatorAgent"],
            llm_config=self.conservative_llm_config,
            human_input_mode= "NEVER"
        )
        # self.consensus_agent = ConversableAgent(
        #     name="ConsensusAgent",
        #     system_message=system_messages["ConsensusAgent"],
        #     llm_config=self.mini4o_llm_config,
        #     human_input_mode= "NEVER"
        # )
        # self.grounding_agent = ConversableAgent(
        #     name="GroundingAgent",
        #     system_message=system_messages["GroundingAgent"],
        #     llm_config=self.mini4o_llm_config,
        #     human_input_mode= "NEVER"
        # )
        self.snap_agent = ConversableAgent(
            name="SnapGuesserAgent",
            system_message=system_messages["SnapGuesserAgent"],
            llm_config=self.snap_llm_config,
            human_input_mode= "NEVER"
        )

    def reset(self):
        """
        Reset the GVCSolver's tracking state for a new game.
        """
        self.reset_agents_state()
        self.failed_guesses.clear()
        self.sorted_failed_guesses = []
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

        for attempt in range(1, self.max_retries + 1):
            self.remaining_str = ', '.join(remaining_words)
            
            # Adding Unsuccessful Guesses
            if len(self.failed_guesses.keys()) > 0:
                # Prepare feedback about unsuccessful and successful categories
                self.feedback = ""
                self.feedback += (
                    "- The following categories represent word groups that have been guessed, but the Game Engine verified "
                    "that these specific groups are not part of the final solution:\n"
                )
                for word_groups in self.sorted_failed_guesses:
                    word_groups_str = ', '.join(word_groups)
                    self.feedback += f"  - {word_groups_str}\n"

            # Adding In previous understandings
            previous_understandings_str = ""
            if self.guesser_past_understandings is not None:
                previous_understandings_str += f"- This is your previous understanding of the board:\n"
                for word_groups in self.guesser_past_understandings:
                    word_groups_str = ', '.join(word_groups) + "\n"
                    previous_understandings_str += f"  * {word_groups_str}"

            # Step 1: GuesserAgent generates a guess and category using remaining words
            # Building Guesser Agent Prompt
            guesser_prompt = ""
            
            if self.feedback:
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
                    f"You last tried to guess {[cat for cat in self.rejected_guesses_buffer]}, and the validator have rejected all of these guess. Do not consider these groupings for the next guess. {self.prev_validator_feedback_if_rejected}\n\n"
                )
                
            guesser_prompt += (
                f"**Remaining Words**\n"
                f"Words: {self.remaining_str}\n"
            )
            
            logger.info(f"GUESSER PROMPT:\n\n{guesser_prompt}")
            logger.info("GuesserAgent: Generating a guess and category.")
            try:
                self.guesser_reply = self._get_agent_reply(self.guesser_agent, guesser_prompt, "GuesserAgent")
                self.last_guess, self.guesser_past_understandings = self.parse_guesser_reply(self.guesser_reply)
                guesser_group, guesser_category = self.last_guess
                guesser_group = [word.strip().upper().replace(",", "") for word in guesser_group]
                guesser_group = sorted(guesser_group)
                logger.info(f"GuesserAgent: Guessed group: {guesser_group} with category: {guesser_category}")
            except ValueError as e:
                logger.error(f"SOLVER: Error parsing GuesserAgent's reply: {e}")
                return (str("Error"), str("Error")), str("Error")
            

            # Step 2: ValidatorAgent validates the category using the entire game board
            grounded, error = self.grounding_check(guesser_group, remaining_words, group_size)
            if len(remaining_words) == group_size:
                if grounded:
                    return tuple(guesser_group), guesser_category
                else:
                    self.rejected_guesses_buffer.append(guesser_group)
                    self.rejected_guesses_buffer = self.insertion_sort_list(self.rejected_guesses_buffer)
                    self.prev_validator_feedback_if_rejected = error
                    logger.info(f"SOLVER: NO Consensus reached for category '{guesser_category}'. Attempt {attempt} of {self.max_retries}.")
            else:
                if grounded: 
                    validator_prompt = (
                        f"**Context:**\n"
                        f"Guesser Agent's reply START: \"\"\"\n{self.guesser_reply}\n\"\"\"\n\nGuesser Agent's reply END\n\n"
                        f"**Remaining Words:**\n"
                        f"Words left on the board: {self.remaining_str}\n\n"  
                        "**Game Engine Feedback**\n\n"
                        f"{self.feedback}\n"
                    )

                    # logger.info(f"VALIDATOR PROMPT:\n\n{validator_prompt}")

                    logger.info("ValidatorAgent: Validating the guess based on the category.")
                    try:
                        self.validator_reply = self._get_agent_reply(self.validator_agent, validator_prompt, "ValidatorAgent")
                        self.validator_dict = self.parse_validator_reply(self.validator_reply)
                    except ValueError as e:
                        logger.error(f"SOLVER: Error parsing ValidatorAgent's reply: {e}")
                        return (str("Error"), str("Error")), str("Error")

                    error = self.validator_dict["validator_feedback"]
                    
                    if self.validator_dict["agreement"]:
                        self.prev_validator_feedback_if_rejected = None
                        # Consensus reached; record successful guess
                        if guesser_category not in self.guesses:
                            self.guesses[guesser_category] = []
                        self.guesses[guesser_category].append(tuple(guesser_group))
                        logger.info(f"SOLVER: Consensus reached for category '{guesser_category}'.")
                        logger.info(f"SOLVER: Attempt {attempt} of {self.max_retries}")
                        return tuple(guesser_group), guesser_category
                
                # Implicie else: Ungrounded
                self.rejected_guesses_buffer.append(guesser_group)
                self.rejected_guesses_buffer = self.insertion_sort_list(self.rejected_guesses_buffer)
                self.prev_validator_feedback_if_rejected = error
                logger.info(f"SOLVER: NO Consensus reached for category '{guesser_category}'. Attempt {attempt} of {self.max_retries}.")
                
        return (str("None"), str("None")), str("None")

    # def grounding_check(self, guess: List[str], remaining_words: List[str], group_size: int) -> bool:
    #     # Rule 1: All words in the guess must be in the Remaining Words list
    #     for word in guess:
    #         if word not in remaining_words:
    #             logger.info(f"Validation Failed: Word '{word}' is not in Remaining Words.")
    #             return False

    #     # Rule 2: The guess must not repeat any grouping in sorted_failed_guesses
    #     if self.sorted_failed_guesses:
    #         sorted_guess = sorted(guess)  # Sort the guess to allow for lexicographical comparison
    #         if sorted_guess in self.sorted_failed_guesses:
    #             logger.info(f"Validation Failed: Guess {sorted_guess} repeats a previously failed grouping.")
    #             return False

    #     # Rule 3: The guess must contain exactly `self.group_size` words
    #     if len(guess) != group_size:
    #         logger.info(f"Validation Failed: Guess {guess} does not contain exactly {group_size} words.")
    #         return False

    #     # If all checks pass, the guess is valid
    #     logger.info(f"Validation Successful: Guess {guess} is valid.")
    #     return True
                
    def grounding_check(self, guess: List[str], remaining_words: List[str], group_size: int) -> Tuple[bool, str]:
        # Preprocess both guess and remaining_words to handle case insensitivity and remove spaces/commas
        print("THE GUESS IS", guess)
        processed_guess = [word.strip().upper().replace(",", "") for word in guess]
        processed_remaining_words = [word.strip().upper().replace(",", "") for word in remaining_words]
        processed_sorted_failed_guesses = [[word.strip().upper().replace(",", "") for word in guess] for guess in self.sorted_failed_guesses]
        error = ""
        # Rule 1: All words in the guess must be in the Remaining Words list
        list_of_wrong_words = []
        for word in processed_guess:
            if word not in processed_remaining_words:
                list_of_wrong_words.append(word)
        if len(list_of_wrong_words) > 0:
            error += f"Validator Disagrees: Word/s '{list_of_wrong_words}' is not in Remaining Words.\n"
            logger.info(f"Validation Failed: Word/s '{list_of_wrong_words}' is not in Remaining Words.")
            return False, error

        # Rule 2: The guess must contain exactly `group_size` words
        if len(processed_guess) != group_size:
            error += f"Validator Disagrees: Guess {processed_guess} does not contain exactly {group_size} words.\n"
            logger.info(f"Validation Failed: Guess {processed_guess} does not contain exactly {group_size} words.")
            return False, error

        # Rule 3: The guess must not repeat any grouping in sorted_failed_guesses
        if len(processed_sorted_failed_guesses) > 0:
            sorted_guess = sorted(processed_guess)  # Sort the guess to allow for lexicographical comparison
            for failed_guesses in processed_sorted_failed_guesses:
                same_words = 0
                for s, g in zip(sorted_guess, failed_guesses): 
                    if s == g:
                        same_words += 1
                if same_words == group_size:   
                    error += f"Validator Disagrees: Guess {sorted_guess} repeats a previously failed grouping.\n"
                    logger.info(f"Validation Failed: Guess {sorted_guess} repeats a previously failed grouping.")
                    return False, error
                else:
                    same_words = 0

        # If all checks pass, the guess is valid
        # logger.info(f"Validation Successful: Guess {processed_guess} is valid.")
        return True, error
                

    def snap_guess(
        self, 
        remaining_words: List[str], 
        entire_game_board: List[str],
        group_size: int = 4, 
        metrics: Optional[Metrics] = None
    ) -> Tuple[Tuple[str, ...], str]:
        
        if metrics is None:
            metrics = Metrics()
                
        self.remaining_str = ', '.join(remaining_words)

        if self.failed_guesses:
            self.feedback = ""
            self.feedback += (
                "Note: You must not return any 4-word groupings from the following 4-word groups as they're not part of the solution:\n"
                # "Note: The below groups are not part of the solution:\n"
            )
            for word_groups in self.sorted_failed_guesses:
                word_groups_str = ', '.join(word_groups)
                self.feedback += f"  - {word_groups_str}\n"

        # Constructing Snap Guesser Prompt
        snap_guesser_prompt = (
            f"Here are some words: {self.remaining_str}\n"  
            f"{self.feedback}\n"
            f"Task: Create one logical grouping that uses 4 words.\n"
        )

        logger.info(f"SNAP PROMPT:\n\n{snap_guesser_prompt}")
        logger.info("SnapGuesserAgent: Generating a guess and category.")

        # Performing the Guess
        try:
            self.guesser_reply = self._get_agent_reply(self.snap_agent, snap_guesser_prompt, "SnapGuesserAgent")
            guesser_group, guesser_category = self.parse_snap_guesser_reply(self.guesser_reply)
            guesser_group = [word.strip().upper().replace(",", "") for word in guesser_group]
            guesser_group = sorted(guesser_group)
            logger.info(f"GuesserAgent: Guessed group: {guesser_group} with category: {guesser_category}")
        except ValueError as e:
            logger.error(f"SOLVER: Error parsing GuesserAgent's reply: {e}")
            return (str("None"), str("None")), str("None")

        grounded, error = self.grounding_check(guesser_group, remaining_words, group_size) 
        if grounded:
            if guesser_category not in self.guesses:
                self.guesses[guesser_category] = []
            self.guesses[guesser_category].append(tuple(guesser_group))
            logger.info(f"SOLVER: Consensus reached for guess '{guesser_group}'")
            return tuple(guesser_group), guesser_category
        else:
            logger.info(f"SOLVER: NO Consensus reached for category '{guesser_category}'.")
        
        return (str("None"), str("None")), str("None")

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
        logger.info(f"{agent_name} raw reply: {reply}")
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

    def parse_guesser_reply(self, reply: str) -> Tuple[Tuple[List[str], str], List[List[str]]]:
        """
        Parse the GuesserAgent's reply to extract the guessed group, category, and overall board understanding.

        :param reply: The raw reply from the GuesserAgent.
        :return: A tuple containing:
                    1. A tuple with the guesser's final guessed group (list of words) and its category (string).
                    2. A dictionary where keys are category descriptions and values are lists of words for each understanding group.
        :raises ValueError: If the reply format is incorrect or required sections are missing.
        """
        try:
            # Normalize the input to remove extra spaces and blank lines
            normalized_reply = re.sub(r"\s*\n\s*", "\n", reply.strip())  # Normalize spaces and newlines
            normalized_reply = re.sub(r"Below are the guesses:.*", "", normalized_reply, flags=re.DOTALL)  # Remove extra text

            # Extract <UNDERSTANDING_OF_BOARD> section
            understanding_section = re.search(
                r"<UNDERSTANDING_OF_BOARD>(.*?)<END_UNDERSTANDING_OF_BOARD>",
                normalized_reply,
                re.DOTALL
            )
            if not understanding_section:
                raise ValueError("Missing <UNDERSTANDING_OF_BOARD> section.")
            # understandings_text = understanding_section.group(1).strip()

            # Regex pattern to parse each group and its category description
            understanding_pattern = re.compile(
                r"Group\d+: (.*?)\\n",
                re.DOTALL
            )

            # Parse groups and categories into a list
            understandings = [match.split(", ") for match in understanding_pattern.findall(understanding_section.group(1))]

            # Extract <GUESS_FOR_THIS_ROUND> section
            guess_section = re.search(
                r"<GUESS_FOR_THIS_ROUND>(.*?)<END_GUESS_FOR_THIS_ROUND>",
                normalized_reply,
                re.DOTALL
            )
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

    def parse_snap_guesser_reply(self, reply: str) -> Tuple[List[str], str]:
        # Regex pattern to extract the reason
        reason_pattern = re.compile(r'"reason":\s*"(.*?)"')
        # Regex pattern to extract the words
        words_pattern = re.compile(r'"words":\s*\[(.*?)\]')

        # Extract reason
        reason_match = reason_pattern.search(reply)
        if not reason_match:
            raise ValueError("Missing 'reason' in the reply.")
        reason = reason_match.group(1)

        # Extract words
        words_match = words_pattern.search(reply)
        if not words_match:
            raise ValueError("Missing 'words' in the reply.")
        
        # Parse words into a list (split by commas and strip quotes/whitespace)
        words_raw = words_match.group(1)
        words = [word.strip().strip('"') for word in words_raw.split(',')]
        
        return words, reason

    # def parse_grounding_reply(self, reply: str) -> Dict[str, Any]:
    #     agreement_match = re.search(r"Agreement to Perform the Guess:\s*(True|False)", reply)
    #     if not agreement_match:
    #         raise ValueError("Missing 'Agreement to Perform the Guess' field.")
    #     agreement = agreement_match.group(1) == "True"
        
    #     return {
    #         "agreement": agreement,
    #     }
        
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

            # # Extract "Rating of correctness"
            # correctness_match = re.search(
            #     r"Rating of guesser agent's correctness of interpreting the board out of \d+:\s*(\d+)", reply
            # )
            # if not correctness_match:
            #     raise ValueError("Missing 'Rating of correctness' field.")
            # correctness_rating = int(correctness_match.group(1))

            # # Extract "Rating of confidence"
            # confidence_match = re.search(
            #     r"Rating of guesser agent's confidence in the guesses out of \d+:\s*(\d+)", reply
            # )
            # if not confidence_match:
            #     raise ValueError("Missing 'Rating of confidence' field.")
            # confidence_rating = int(confidence_match.group(1))

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
                # "correctness_rating": correctness_rating,
                # "confidence_rating": confidence_rating,
                "validator_feedback": validator_feedback
            }

        except Exception as e:
            raise ValueError(f"Error parsing validation report: {str(e)}")

    def insertion_sort_list(self, lst):
        """
        Sort a list in ascending order using the insertion sort algorithm.

        Args:
            lst (List): The list to sort.
        
        Returns:
            List: A sorted version of the input list.
        """
        for i in range(1, len(lst)):
            key = lst[i]
            j = i - 1

            # Compare key with each element on the left until it finds the proper position
            while j >= 0 and key < lst[j]:
                lst[j + 1] = lst[j]
                j -= 1

            # Place the key in its correct location
            lst[j + 1] = key

        return lst

    def reset_agents_state(self):
        self.guesser_past_understandings = None
        self.last_guess = None
        self.rejected_guesses_buffer = deque(maxlen=self.max_retries)
        self.prev_validator_feedback_if_rejected = None
        self.validator_dict = {}
        self.guesser_reply = None
        self.validator_reply = None
        self.remaining_str = None
        self.feedback = None
        self.snap_correct = False

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
        error_counter = 0
        wrong_counter = 0
        
        # Initialize Agents
        self.initialize_agents(self.get_prompts(game.group_size))
        
        # Conservative Guessing
        while not game.is_over:
            
            error_counter = 0
            wrong_counter = 0
            logger.info("SOLVER: Conservative Guessing")
            
            while not game.is_over:
                try:
                    remaining_words = game.all_words  # Current remaining words
                    guess, reasoning = self.guess(
                        remaining_words=remaining_words,
                        entire_game_board=entire_game_board,
                        group_size=game.group_size,
                        metrics=metrics
                    )
                    
                    if reasoning == str("Error"): # Errored out, reset and retry
                        self.reset_agents_state()
                        error_counter += 1
                        if error_counter == self.max_conservative_round_errors:
                            self.reset_agents_state()
                            break
                        continue
                    if reasoning == str("None"): # No Consensus after internal looping
                        self.reset_agents_state()
                        break
                    
                    # Attempt to check the guess
                    cat = game.category_guess_check(list(guess))
                    logger.info(f"GAME ENGINE: Guessed: {guess} --> {cat}")
        
                    if cat is None: # If the guess is wrong
                        previous_guesses.add(tuple(guess))
                        metrics.hallucination_words(list(guess), remaining_words)
                        metrics.increment_failed_guesses()
                        self.failed_guesses[reasoning] = guess
                        self.sorted_failed_guesses.append(sorted(guess))
                        self.sorted_failed_guesses = self.insertion_sort_list(self.sorted_failed_guesses)
                        wrong_counter += 1
                    else: # If the guess is correct
                        guessed_cat_idx = game._og_groups.index(cat)
                        metrics.add_solve(level=guessed_cat_idx)
                        metrics.cosine_similarity_category(guessed_cat=reasoning, correct_cat=cat.group)
                        wrong_counter = 0 # Reset if Correct
                except GameOverException as e:
                    logger.warning(str(e))
                    break
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    break
                
                # "Reset the State of the agents"
                self.reset_agents_state()
                
                if wrong_counter >= self.max_conservative_wrong_guesses:
                    # self.reset_agents_state()
                    break
                
            logger.info("SOLVER: Snap Guessing")

            # Snap Guessing until the next correct guess
            self.snap_correct = False
            while not game.is_over and not self.snap_correct:
                try:
                    remaining_words = game.all_words  # Current remaining words
                    guess, reasoning = self.snap_guess(
                        remaining_words=remaining_words,
                        entire_game_board=entire_game_board,
                        group_size=game.group_size,
                        metrics=metrics
                    )
                    
                    if reasoning == str("None"):
                        continue
                    
                    # Attempt to check the guess
                    cat = game.category_guess_check(list(guess))
                    logger.info(f"GAME ENGINE: Guessed: {guess} --> {cat}")
        
                    if cat is None: # If the guess is wrong
                        previous_guesses.add(tuple(guess))
                        metrics.hallucination_words(list(guess), remaining_words)
                        metrics.increment_failed_guesses()
                        self.failed_guesses[reasoning] = guess
                        self.sorted_failed_guesses.append(sorted(guess))
                        self.sorted_failed_guesses = self.insertion_sort_list(self.sorted_failed_guesses)
                    else: # If the guess is correct
                        guessed_cat_idx = game._og_groups.index(cat)
                        metrics.add_solve(level=guessed_cat_idx)
                        metrics.cosine_similarity_category(guessed_cat=reasoning, correct_cat=cat.group)
                        self.snap_correct = True
                        break
                except GameOverException as e:
                    logger.warning(str(e))
                    break
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    break
                
                # "Reset the State of the agents"
                self.reset_agents_state()

        if commit_to:
            metrics.commit(to_db=commit_to)
        
        # Reset all agent states
        self.reset()    
        
        return game.solved_categories
