# src/hw_emulator.py
import numpy as np
import random
from src.cs_utils import compress, recover_bp, quality_metric
from src.lora_phy import energy_tx

def load_channel_traces(num_nodes=20, trace_file=None):
    """
    Emulate channel traces: returns list of (SNR, fading_type) for each node.
    If trace_file is None, generate synthetic traces.
    """
    if trace_file is None:
        # Synthetic traces: random SNR between -20 and +20 dB
        traces = []
        for _ in range(num_nodes):
            snr = np.random.uniform(-20, 20)
            fading = np.random.choice(['AWGN', 'Rayleigh', 'Rician'], p=[0.4, 0.3, 0.3])
            traces.append((snr, fading))
        return traces
    else:
        # For reproducibility, we could read from a CSV file
        # For now, we use the synthetic version
        return load_channel_traces(num_nodes)

def emulate_transmission(signal, q_min, params, snr_db, fading_type):
    """
    Emulate a single transmission: apply CS, channel, recovery, return energy and quality.
    """
    sf, cr, cf = params
    N = len(signal)
    M = max(1, int(cf * N))
    Phi = generate_sensing_matrix(M, N, seed=random.randint(0, 100000))
    y = compress(signal, Phi)
    # Add noise based on SNR
    sigma = np.std(y) * 10 ** (-snr_db / 20)
    # For simplicity, we only add AWGN; fading can be modeled as scale factor
    if fading_type == 'Rayleigh':
        h = np.random.rayleigh(1, M)
        y = y * h
    elif fading_type == 'Rician':
        K = 3  # K-factor
        h = np.sqrt(2*K/(K+1)) + np.random.randn(M) + 1j*np.random.randn(M)  # simplified
        h = np.abs(h)  # use magnitude
        y = y * h
    y_noisy = y + np.random.randn(M) * sigma
    x_hat = recover_bp(y_noisy, Phi, epsilon=0.1*sigma)
    q = quality_metric(signal, x_hat)
    if q < q_min:
        return None, None  # quality violation
    payload_bytes = int(np.ceil(M / 8))
    e = energy_tx(sf, cr, payload_bytes)
    return e, q