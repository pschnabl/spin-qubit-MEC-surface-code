from sinter import TaskStats
import sinter
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(stats: List[sinter.TaskStats], ax=None, logical_error_per_round=False):
    """
    Plot the logical error rate vs physical error rate for a list of TaskStats.
    Adds a different linestyle for X and Z memories / V and H memories.
    Lets the user choose between logical error per shot or per round.

    Args:
        stats (List[sinter.TaskStats]):
            List of sinter.TaskStats.
        ax (optional):
            axis to plot on. Defaults to None.
        logical_error_per_round (bool, optional):
            If True the logical error per round is plotted instead of the logical error per shot. Defaults to False.
    """
    def custom_plot_args_func(
        curve_index: int,  # a unique incrementing integer for each curve
        curve_group_key: Any,  # what group_func returned
        stats: List[sinter.TaskStats],  # the data points on the curve
    ) -> Dict[str, Any]:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # can be used to assign different colors to different distances
        plot_args = {}
        for stat in stats:
            if stat.json_metadata['memory'] == "X" or stat.json_metadata['memory'] == "V":
                plot_args['linestyle'] = '-'
            elif stat.json_metadata['memory'] == "Z" or stat.json_metadata['memory'] == "H":
                plot_args['linestyle'] = '--'            
        return plot_args
    
    if logical_error_per_round:
        failure_units_per_shot_func=lambda stat: stat.json_metadata["params"]["rounds"]
    else:
        failure_units_per_shot_func=lambda _: 1
        
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        group_func = lambda stat: f"d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata['p'],
        plot_args_func=custom_plot_args_func,
        failure_units_per_shot_func=failure_units_per_shot_func,
    )
    ax.loglog()
    ax.grid()
    ax.set_title('Logical Error Rate vs Physical Error Rate')
    ax.set_ylabel('Logical Error Probability ' + [' (per shot)', ' (per round)'][logical_error_per_round])
    ax.set_xlabel('Physical Error Rate')
    ax.legend()
    

def calculate_logical_error_rate(task_stats: TaskStats, per_round=False) -> float:
    """
    Calculate the logical error rate from TaskStats.
    If per_round is True, the logical error rate is calculated per round.
    """
    per_shot = task_stats.errors / task_stats.shots
    if per_round:
        return sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=task_stats.json_metadata["params"]["rounds"])
    else:
        return per_shot

def extract_data_from_stats(task_stats_list: list[TaskStats], per_round=False) -> list[tuple[int, float, float]]:
    """
    Extract the distance, physical error rate, and logical error rate from a list of TaskStats.
    """
    data = []
    for stats in task_stats_list:
        distance = stats.json_metadata['d']
        physical_error_rate = stats.json_metadata['p']
        logical_error_rate = calculate_logical_error_rate(stats, per_round)
        data.append((distance, physical_error_rate, logical_error_rate))
    return data

def estimate_threshold(task_stats_list: list[TaskStats], logical_error_per_round=False) -> float:
    """
    Estimate the threshold error rate from a list of TaskStats.
    """
    data = extract_data_from_stats(task_stats_list, logical_error_per_round)
    
    # Group data by physical error rate
    error_rate_groups = {}
    for distance, error_rate, logical_error_rate in data:
        if error_rate not in error_rate_groups:
            error_rate_groups[error_rate] = []
        error_rate_groups[error_rate].append((distance, logical_error_rate))
    
    # Find the physical error rate where logical error rates for different distances are close
    threshold_estimates = []
    for error_rate, group in error_rate_groups.items():
        distances, logical_error_rates = zip(*group)
        # Check the normalized variance or spread of logical error rates across distances
        if len(logical_error_rates) > 1:
            spread = max(logical_error_rates) - min(logical_error_rates)
            threshold_estimates.append((error_rate, spread/min(logical_error_rates)))
    
    # Select the error rate with the smallest spread
    estimated_threshold = min(threshold_estimates, key=lambda x: x[1])[0]
    return estimated_threshold


## for the biased noise model in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.13.031007 the x-axis has to be rescaled
def plot_stats_bias(stats: List[sinter.TaskStats], ax=None):
    def custom_plot_args_func(
        curve_index: int,  # a unique incrementing integer for each curve
        curve_group_key: Any,  # what group_func returned
        stats: List[sinter.TaskStats],  # the data points on the curve
        ) -> Dict[str, Any]:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plot_args = {}
        for stat in stats:
            if stat.json_metadata['memory'] == "X":
                plot_args['linestyle'] = '-'
            elif stat.json_metadata['memory'] == "Z":
                plot_args['linestyle'] = '--'
            if stat.json_metadata['d'] == 5:
                plot_args['color'] = colors[0]
                # plot_args['marker'] = 'x'
            elif stat.json_metadata['d'] == 7:
                plot_args['color'] = colors[1]
                # plot_args['marker'] = 'o'
            elif stat.json_metadata['d'] == 9:
                plot_args['color'] = colors[2]
                # plot_args['marker'] = 's'
            elif stat.json_metadata['d'] == 11:
                plot_args['color'] = colors[3]
            elif stat.json_metadata['d'] == 13:
                plot_args['color'] = colors[4]
        return plot_args
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        # group_func=lambda stat: f"d={stat.json_metadata['d']}, eta={stat.json_metadata['eta']}",
        # x_func=lambda stat: stat.json_metadata['p'],
        group_func = lambda stat: f"d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata['p'] * (1/5 + (4 / (5*stat.json_metadata['eta']) )),
        # plot_args_func= lambda curve_index, curve_group_key, stats: {"linestyle": "--" if "X" in curve_group_key else "-.", "alpha": 0.5 if "X" in curve_group_key else 1},
        plot_args_func=custom_plot_args_func,
        # line_fits=("log","log")
    )
    ax.loglog()
    ax.grid()
    ax.set_title('Logical Error Rate vs Physical Error Rate')
    ax.set_ylabel('Logical Error Probability (per shot)')
    ax.set_xlabel(r'CNOT infidelity: $p \cdot (1/5 + 4/(5\eta))$') # pCX = p * (1/5 + 4/(5*eta)) see: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.13.031007
    ax.legend()

def extract_data_from_stats_bias(task_stats_list: list[TaskStats]) -> list[tuple[int, float, float]]:
    """
    Extract the distance, physical error rate, and logical error rate from a list of TaskStats.
    """
    data = []
    for stats in task_stats_list:
        distance = stats.json_metadata['d']
        physical_error_rate = stats.json_metadata['p'] * (1/5 + (4 / (5*stats.json_metadata['eta']) )) #stats.json_metadata['p']/3 * (2/stats.json_metadata['eta'] + 1)
        logical_error_rate = calculate_logical_error_rate(stats)
        data.append((distance, physical_error_rate, logical_error_rate))
    return data

def estimate_threshold_bias(task_stats_list: list[TaskStats]) -> float:
    """
    Estimate the threshold error rate from a list of TaskStats.
    """
    data = extract_data_from_stats_bias(task_stats_list)
    
    # Group data by physical error rate
    error_rate_groups = {}
    for distance, error_rate, logical_error_rate in data:
        if error_rate not in error_rate_groups:
            error_rate_groups[error_rate] = []
        error_rate_groups[error_rate].append((distance, logical_error_rate))
    
    # Find the physical error rate where logical error rates for different distances are close
    threshold_estimates = []
    for error_rate, group in error_rate_groups.items():
        distances, logical_error_rates = zip(*group)
        # Check the normalized variance or spread of logical error rates across distances
        if len(logical_error_rates) > 1:
            spread = max(logical_error_rates) - min(logical_error_rates)
            threshold_estimates.append((error_rate, spread/min(logical_error_rates)))
    
    # Select the error rate with the smallest spread
    estimated_threshold = min(threshold_estimates, key=lambda x: x[1])[0]
    return estimated_threshold