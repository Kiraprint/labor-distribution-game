from matplotlib import pyplot as plt
import numpy as np

def plot_profit_distribution(profit_data, title="Profit Distribution"):
    plt.figure(figsize=(10, 6))
    for coalition, profits in profit_data.items():
        plt.plot(profits, label=f'Coalition {coalition}')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid()
    plt.show()

def plot_agent_performance(agent_performance_data, title="Agent Performance"):
    plt.figure(figsize=(10, 6))
    for agent, performance in agent_performance_data.items():
        plt.plot(performance, label=f'Agent {agent}')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Performance Metric')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_simulation_results(profit_data, agent_performance_data):
    plot_profit_distribution(profit_data)
    plot_agent_performance(agent_performance_data)