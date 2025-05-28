def calculate_average_profit(profits):
    return sum(profits) / len(profits) if profits else 0

def calculate_variance(profits):
    average = calculate_average_profit(profits)
    return sum((x - average) ** 2 for x in profits) / len(profits) if profits else 0

def calculate_shapley_value(contributions, total_profit):
    shapley_values = {}
    for agent, contribution in contributions.items():
        shapley_values[agent] = contribution / total_profit if total_profit > 0 else 0
    return shapley_values

def calculate_owen_value(contributions, total_profit):
    owen_values = {}
    for agent, contribution in contributions.items():
        owen_values[agent] = contribution / total_profit if total_profit > 0 else 0
    return owen_values

def evaluate_agent_performance(agent_profits):
    average_profit = calculate_average_profit(agent_profits)
    profit_variance = calculate_variance(agent_profits)
    return {
        'average_profit': average_profit,
        'profit_variance': profit_variance
    }