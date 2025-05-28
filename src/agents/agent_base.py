class AgentBase:
    def __init__(self, agent_id, level):
        self.agent_id = agent_id
        self.level = level
        self.resources = 0
        self.news = []
        self.coalition = None

    def generate_news(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def share_resources(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def receive_news(self, news):
        self.news.append(news)

    def form_coalition(self, other_agents):
        self.coalition = other_agents

    def calculate_profit(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self):
        self.resources = 0
        self.news = []
        self.coalition = None