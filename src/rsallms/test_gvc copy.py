
from autogen import ConversableAgent, GroupChat, GroupChatManager
class GVCSolver():
    def __init__(self):
        self.initialize_agents()
        
    def initialize_agents(self):

        self.llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

        # Create agents using ConversableAgent
        self.guesser_agent = ConversableAgent(
            name="GuesserAgent",
            system_message="You are a Guesser Agent in a word game. Given a list of words, propose a group of 4 related words and a corresponding category.",
            llm_config=self.llm_config,
        )

        self.validator_agent = ConversableAgent(
            name="ValidatorAgent",
            system_message="You are a Validator Agent. Given a list of words and a category, identify a group of 4 words that fit the category.",
            llm_config=self.llm_config,
        )

        self.consensus_agent = ConversableAgent(
            name="ConsensusAgent",
            system_message="You are a Consensus Agent. Compare two groups of words and determine if they match.",
            llm_config=self.llm_config,
        )
    
    def run(self):
        # Groupchat to provide the lisst of agents
        group_chat = GroupChat(
            agents=[self.guesser_agent, self.validator_agent, self.consensus_agent],
            messages=[],
            max_round=3,
            speaker_selection_method="round_robin",
            #  the next speaker is selected in a round robin fashion, i.e., iterating in the same order as provided in agents.
        )
        
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
        )
        
def main():
    solver = GVCSolver()
    solver.run()

if __name__ == "__main__":
    main()
