import unittest
from src.agents.level1_agent import Level1Agent
from src.agents.level2_agent import Level2Agent
from src.environment.coalition import Coalition

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.level1_agent = Level1Agent(agent_id=1)
        self.level2_agent = Level2Agent(agent_id=2)
        self.coalition = Coalition()

    def test_level1_agent_initialization(self):
        self.assertEqual(self.level1_agent.agent_id, 1)
        self.assertIsNotNone(self.level1_agent.resources)

    def test_level2_agent_initialization(self):
        self.assertEqual(self.level2_agent.agent_id, 2)
        self.assertIsNotNone(self.level2_agent.hub_connections)

    def test_level1_agent_news_generation(self):
        news = self.level1_agent.generate_news()
        self.assertIsInstance(news, dict)
        self.assertIn('news_type', news)
        self.assertIn('content', news)

    def test_level2_agent_strategy_selection(self):
        strategy = self.level2_agent.select_strategy()
        self.assertIn(strategy, ['cooperative', 'competitive'])

    def test_coalition_profit_distribution(self):
        self.coalition.add_agent(self.level1_agent)
        self.coalition.add_agent(self.level2_agent)
        profit = self.coalition.calculate_profit()
        self.assertGreaterEqual(profit, 0)

    def test_level1_agent_resource_distribution(self):
        initial_resources = self.level1_agent.resources
        self.level1_agent.distribute_resources()
        self.assertNotEqual(self.level1_agent.resources, initial_resources)

if __name__ == '__main__':
    unittest.main()