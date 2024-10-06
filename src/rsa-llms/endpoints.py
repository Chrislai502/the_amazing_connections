
import requests

from dataclasses import dataclass

@dataclass
class Endpoint:
    """
    Simple interface for interacting with an OAI API endpoint.

    :param base_url: the base URL of the API
    :param model: the model to use for the chat completion endpoint
    """

    base_url: str
    model: str

    CHAT_COMPLETION = "v1/chat/completions"

    @property
    def chat_url(self):
        return f"{self.base_url}/{Endpoint.CHAT_COMPLETION}"

    def respond(self, message: str, system_promt: str | None = None) -> str:
        headers = {"Content-Type": "application/json"}

        messages = [ {"role": "user", "content": message} ]
        if system_promt is not None:
            messages.insert(0, {"role": "system", "content": system_promt})

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        response = requests.post(self.chat_url, headers=headers, json=data)

        try:
            json_response = response.json()
        except:
            print(response.text)

        return json_response['choices'][0]['message']['content']
