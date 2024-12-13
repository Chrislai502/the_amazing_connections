
import importlib.resources

from typing import TypeAlias, Callable
from dataclasses import dataclass
from typing import Callable
from os import environ as env
import time

import chevron
import requests
import re
from datetime import timedelta

import dotenv
try:
    dotenv.load_dotenv()
except:
    print(f"Could not load environment variables. Continuing without them ...")

from .metrics import Metrics

PROMPTS_FOLDER = importlib.resources.files("rsallms").joinpath("prompts")
EndpointConfig: TypeAlias = dict[str, "Endpoint"]


@dataclass
class Endpoint:
    """
    Common interface for interacting with an OAI API endpoint (e.g. ollama, groq, etc.)

    :param base_url: the base URL of the API
    :param model: the model to use for the chat completion endpoint
    :param api_key: (optional) the api key necessary to complete requests
    """

    DEFAULTS = {
        "oai": {
            "base_url": "https://api.openai.com/",
            "api_key": "OPENAI_API_KEY"
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/",
            "api_key": "GROQ_API_KEY"
        }
    }

    base_url: str
    """
    The base URL for the api. Can also be a key in `Endpoint.DEFAULTS`
    as shorthand for a url and to override the `api_key`. This shorthand
    should not be used in conjunction with `api_key`.
    """

    model: str
    """The name of the model provided by the endpoint"""

    api_key: str | None = None
    """[Optional] The API key required to establish a connection"""

    CHAT_COMPLETION = "v1/chat/completions"

    def __post_init__(self):
        # resolve commonly used endpoints
        if self.base_url in Endpoint.DEFAULTS:
            info = Endpoint.DEFAULTS[self.base_url]
            self.base_url = info["base_url"]

            api_key_env_var = info["api_key"]
            if api_key_env_var not in env:
                raise OSError(f"API Key {api_key_env_var} not found!")
            self.api_key = env[api_key_env_var]

    @property
    def chat_url(self):
        return f"{self.base_url}/{Endpoint.CHAT_COMPLETION}"

    def respond(self, message: str, system_prompt: str | None = None, temperature: float | None = None, metrics: Metrics | None = None, retries: int = 1) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if temperature is None:
            temperature = 0.7
        messages = [{"role": "user", "content": message}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = requests.post(self.chat_url, headers=headers, json=data)

        try:
            json_response = response.json()
        except Exception as e:
            raise Exception(response.text) from e

        if 'error' in json_response:
            if 'retry-after' in response.headers:
                retry_after = int(response.headers['retry-after'])
                time.sleep(retry_after)
                return self.respond(message, system_prompt, temperature, metrics, retries)
            elif 'x-ratelimit-reset-requests' in response.headers:  # time until rate limit resets for requests
                reset_time = response.headers['x-ratelimit-reset-requests']
                parsed_time = re.match(r'(?:(\d+)m)?([\d.]+)s', reset_time)
                if not parsed_time:
                    raise ValueError(
                        f"Invalid time format in header: {reset_time}")
                minutes = int(parsed_time.group(1) or 0) # in case no minutes
                seconds = float(parsed_time.group(2))
                sleep_time = timedelta(minutes=minutes, seconds=seconds).total_seconds()
                return self.respond(message, system_prompt, temperature, metrics, retries)
            elif 'x-ratelimit-reset-tokens' in response.headers: # time until rate limit resets for tokens
                reset_time = response.headers['x-ratelimit-reset-tokens']
                parsed_time = re.match(r'(?:(\d+)m)?([\d.]+)s', reset_time)
                if not parsed_time:
                    raise ValueError(
                        f"Invalid time format in header: {reset_time}")
                minutes = int(parsed_time.group(1) or 0) # in case no minutes
                seconds = float(parsed_time.group(2))
                sleep_time = timedelta(minutes=minutes, seconds=seconds).total_seconds()
                time.sleep(sleep_time)
                return self.respond(message, system_prompt, temperature, metrics, retries)
            else:
                print(response.headers)
                raise ValueError(
                    f"Error in endpoint request!: {json_response['error']}")
        if 'choices' not in json_response:
            raise ValueError(
                f"Malformed response from endpoint!: Got: {json_response}")

        if metrics is not None:
            metrics.add_tokens(
                self.model,
                prompt_tokens=json_response['usage']['prompt_tokens'],
                completion_tokens=json_response['usage']['completion_tokens']
            )
        return json_response['choices'][0]['message']['content']


class CannedResponder(Endpoint):
    def __init__(self, responder_func: Callable[[str, str | None], str]):
        super().__init__("", "")
        self.responder = responder_func

    def respond(self, message, system_prompt=None, temperature=None, metrics=None, retries=1):
        return self.responder(message, system_prompt)


def get_prompt(name: str, **kwargs) -> str:
    with PROMPTS_FOLDER.joinpath(f"{name}.mustache").open() as f:
        return chevron.render(f.read(), data=kwargs).strip()


def generate_prompt(all_words: list[str], category: str | None, num_shots: int, type: str = "multi_shot_prompt") -> str:
    examples = prepare_examples(num_shots, include_category=category is not None)
    prompt = get_prompt(
        name=type,
        instructions={'num_words': len(all_words)},
        examples=examples,
        current_words=', '.join(all_words),
        current_category=category  # This can be None
    )
    return prompt

def prepare_examples(num_shots: int, include_category: bool = True) -> list[dict]:
    """
    Prepare a list of examples for multi-shot prompting.

    :param num_shots: Number of examples to include.
    :param include_category: Whether to include categories in the examples.
    :return: A list of example dictionaries.
    """
    all_examples = [
        {
            'words': 'Bass, Flounder, Salmon, Trout, Ant, Drill, Island, Opal',
            'category': 'types of fish' if include_category else None,
            'response': '{"groups": [{"reason": "types of fish", "words": ["Bass", "Flounder", "Salmon", "Trout"]}]}'
        },
        {
            'words': 'Ant, Drill, Island, Opal, Bass, Flounder, Salmon, Trout',
            'category': 'things that start with FIRE' if include_category else None,
            'response': '{"groups": [{"reason": "things that start with FIRE", "words": ["Ant", "Drill", "Island", "Opal"]}]}'
        },
        {
            'words': 'ALLEY, DRIVE, LANE, STREET, BLISS, CLOUD NINE, HEAVEN, PARADISE',
            'category': 'ROAD NAMES' if include_category else None,
            'response': '{"groups": [{"reason": "ROAD NAMES", "words": ["ALLEY", "DRIVE", "LANE", "STREET"]}]}'
        },
        {
            'words': 'BLISS, CLOUD NINE, HEAVEN, PARADISE, ALLEY, DRIVE, LANE, STREET',
            'category': 'STATES OF ELATION' if include_category else None,
            'response': '{"groups": [{"reason": "STATES OF ELATION", "words": ["BLISS", "CLOUD NINE", "HEAVEN", "PARADISE"]}]}'
        },
        {
            'words': 'CIRCUS, SATURN, TREE, WEDDING, BLISS, CLOUD NINE, HEAVEN, PARADISE',
            'category': 'THINGS WITH RINGS' if include_category else None,
            'response': '{"groups": [{"reason": "THINGS WITH RINGS", "words": ["CIRCUS", "SATURN", "TREE", "WEDDING"]}]}'
        },
    ]
    examples = all_examples[:num_shots]

    return examples



def chain_prompts(files: list[str], **kwargs) -> str:
    content = []
    for file in files:
        content.append(get_prompt(file, **kwargs))
    return "\n".join(content)
