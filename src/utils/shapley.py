def calculate_shapley_value(coalition_values, num_agents):
    shapley_values = [0] * num_agents
    for agent in range(num_agents):
        for coalition in range(1 << num_agents):
            if coalition & (1 << agent):
                continue
            coalition_value = coalition_values[coalition | (1 << agent)] - coalition_values[coalition]
            num_permutations = factorial(bin(coalition).count('1'))
            shapley_values[agent] += coalition_value * num_permutations / (num_agents * factorial(num_agents - 1))
    return shapley_values

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def calculate_shapley_for_coalitions(coalition_values):
    num_agents = len(coalition_values[0])
    shapley_values = calculate_shapley_value(coalition_values, num_agents)
    return shapley_values