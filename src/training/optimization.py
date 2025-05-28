from typing import List, Tuple, Callable
import numpy as np

def optimize_hyperparameters(param_grid: dict, evaluate: Callable) -> Tuple[dict, float]:
    best_params = {}
    best_score = float('-inf')

    for params in parameter_combinations(param_grid):
        score = evaluate(params)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def parameter_combinations(param_grid: dict) -> List[dict]:
    from itertools import product

    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    return combinations

def normalize_rewards(rewards: List[float]) -> List[float]:
    mean = np.mean(rewards)
    std = np.std(rewards)
    if std == 0:
        return rewards
    return [((r - mean) / std).item() for r in rewards]

def update_learning_rate(optimizer, epoch: int, initial_lr: float, decay: float) -> None:
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def early_stopping(validation_scores: List[float], patience: int) -> bool:
    if len(validation_scores) < patience:
        return False
    return all(validation_scores[i] <= validation_scores[i + 1] for i in range(-patience, -1))