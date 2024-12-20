# Source: https://microsoft.github.io/autogen/0.2/docs/notebooks/agentchat_custom_model#create-and-configure-the-custom-model
# Source: https://microsoft.github.io/autogen/0.2/blog/2024/01/26/Custom-Models
# https://github.com/microsoft/autogen/blob/0.2/notebook/agentchat_custom_model.ipynb
# pip install autogen-agentchat~=0.2

from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from .endpoints import Endpoint, get_prompt, EndpointConfig
ENDPOINTS: EndpointConfig = {
    "default": Endpoint(
        "groq",
        # model="llama-3.2-1b-preview",  # this is 4 cents per Mil. tok, i.e. free
        # model="llama-3.2-3b-preview",
        # model="llama-3.2-90b-vision-preview",
        model="llama-3.1-70b-versatile"
    )
}
class CustomModelClient:
    def __init__(self, config, **kwargs):
        pass
        # Print the configuration details for debugging purposes
        # print(f"CustomModelClient config: {config}")

        # # Set the device (CPU or GPU) based on the config, defaulting to "cpu" if not specified
        # self.device = config.get("device", "cpu")

        # # Load the specified model for causal language modeling and move it to the specified device
        # # `AutoModelForCausalLM` is a transformer model class that supports text generation
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     config["model"]).to(self.device)
        # print("MODELLLL:", type(self.model))

        # # Store the model name for future reference
        # self.model_name = config["model"]

        # # Load the tokenizer associated with the model (for encoding/decoding text to/from tokens)
        # # `use_fast=False` specifies using a slower but potentially more compatible tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     config["model"], use_fast=False)

        # # Set the padding token ID to the same value as the end-of-sequence (eos) token ID
        # # This ensures the model correctly handles padding when generating responses
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # # Retrieve generation-specific parameters from the config, if any, with defaults
        # # The maximum length of generated responses is set to 256 if not specified
        # gen_config_params = config.get("params", {})
        # self.max_length = gen_config_params.get("max_length", 256)

        # # Print a message confirming the model and device setup
        # print(f"Loaded model {config['model']} to {self.device}")

    def create(self, params):
        # Check if streaming is requested; raise an error if so, as it's not supported locally
        # Streaming responses (sending partial responses in real-time) is not implemented here
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("REST API models do not support streaming.")
        else:
            # Determine the number of responses to generate; defaults to 1
            num_of_responses = params.get("n", 1)

            # Initialize a response object using SimpleNamespace for flexible attribute access
            # This object will hold the model's responses in a compatible format
            response = SimpleNamespace()

            # Testing
            json_response = ENDPOINTS["default"].test_respond(message=params["messages"][1]['content'], system_prompt=params["messages"][0]['content'])

            # Crafting the response object
            response.choices = []
            for _ in range(num_of_responses):
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = json_response
                choice.message.function_call = None

                response.choices.append(choice)
            # Return the response object, which contains the generated text(s) in response.choices
            return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        # Extracts the content of each generated message from the response object
        # Returns a list of text responses, one for each generated choice
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        # Set the response cost to 0, as no real costing function is implemented
        # This might be useful if tracking token usage costs, but here it always returns 0
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        # Returns a dictionary with usage statistics, like number of tokens used
        # Currently, returns an empty dictionary. If usage tracking is needed, this
        # function could return prompt/completion tokens, total tokens, cost, model, etc.
        return {}
