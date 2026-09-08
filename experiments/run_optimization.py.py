# experiments/run_optimization.py
import numpy as np
import os
from src.nsga2_optimizer import run_nsga2
from data.generate_signals import generate_synthetic_signal

def main():
    # Generate a fixed signal
    signal = generate_synthetic_signal(N=1024, freqs=[5,12,25], amps=[1.0,0.5,0.3])
    snr_list = np.arange(-20, 21, 5)
    results = {}
    for snr in snr_list:
        print(f"Running NSGA-II for SNR = {snr} dB")
        front, _, _ = run_nsga2(signal, snr, pop_size=100, ngen=200)
        results[snr] = front
        np.save(f"pareto_front_snr_{snr}.npy", front)
    print("✅ Optimization completed. Pareto fronts saved.")
    # Example: print front for SNR=0
    print("First 5 Pareto points for SNR=0:", results[0][:5])

if __name__ == "__main__":
    main()