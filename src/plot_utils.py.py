# src/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_front(front, title='Pareto Front', save_path=None):
    energies = [pt[0] for pt in front]
    qualities = [pt[1] for pt in front]
    plt.figure(figsize=(6,4))
    plt.scatter(energies, qualities, c='blue', alpha=0.7)
    plt.xlabel('Energy [mJ]')
    plt.ylabel('Quality Q')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_orchestration_comparison(results, save_path=None):
    """
    results: dict with keys 'CO', 'DO', 'HO' each a dict with 'energy', 'quality', 'pdr'
    """
    labels = list(results.keys())
    energies = [results[k]['energy'] for k in labels]
    qualities = [results[k]['quality'] for k in labels]
    pdrs = [results[k]['pdr'] for k in labels]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x - width, energies, width, label='Energy [mJ]')
    ax.bar(x, qualities, width, label='Quality Q')
    ax.bar(x + width, pdrs, width, label='PDR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True, axis='y')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()