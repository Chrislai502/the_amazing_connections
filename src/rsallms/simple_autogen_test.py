from autogen import ConversableAgent
from autogen.function_utils import get_function_schema
# lass AssistantAgent(ConversableAgent) https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/assistant_agent#assistantagent
from autogen import AssistantAgent, UserProxyAgent
from rsallms import (
    CustomModelClient,
    NaiveSolver
)

import autogen

# Set up and register Config for the custom model
config_list_custom = autogen.config_list_from_json(
    env_or_file="autogen_agents.json",
    filter_dict={"model_client_cls": ["CustomModelClient"]},
)

# Get the actual config
llm_config = {"config_list": config_list_custom}
entrypoint_agent_system_message = "You are an agent, just be nice :)"


# # Testing outputs of the current model
# solver = NaiveSolver()
# response = solver.test_response()
# print(response)
# exit(0)

# Set up and register ModelClient for the custom model
assistant = AssistantAgent("assistant", llm_config=llm_config)
assistant.register_model_client(model_client_cls=CustomModelClient)
assistant.initiate_chat(
    assistant, message="Write python code to print Hello World!")
