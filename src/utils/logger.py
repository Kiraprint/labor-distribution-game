import logging
import os
from datetime import datetime


def setup_logger():
    """Set up the logger for the simulation"""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/simulation_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger("labor_game")


# Create specialized loggers
def get_agent_logger(agent_id):
    return logging.getLogger(f"labor_game.agent.{agent_id}")


def get_game_logger():
    return logging.getLogger("labor_game.game")


def get_coalition_logger():
    return logging.getLogger("labor_game.coalition")
