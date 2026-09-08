import numpy as np
import os

def generate_synthetic_signal(N=1024, freqs=[5, 12, 25], amps=[1.0, 0.5, 0.3], seed=42):
    """Generate a synthetic vibration signal (sum of sinusoids)."""
    np.random.seed(seed)
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    # Add slight noise
    x += 0.05 * np.random.randn(N)
    return x

if __name__ == "__main__":
    # Generate multiple signals for dataset
    os.makedirs("data", exist_ok=True)
    signals = [generate_synthetic_signal(seed=i) for i in range(10)]
    np.save("data/signals.npy", np.array(signals))
    print("✅ Synthetic signals saved to data/signals.npy")