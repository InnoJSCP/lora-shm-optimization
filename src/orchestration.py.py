import numpy as np
import random
from src.cs_utils import generate_sensing_matrix, compress, recover_bp, quality_metric
from src.lora_phy import energy_tx

def greedy_local_search(signal, noise_snr_db, q_min):
    """Distributed greedy local search."""
    best_e = float('inf')
    best_params = None
    cf_candidates = np.linspace(0.2, 1.0, 10)
    for sf in [7,8,9,10,11,12]:
        for cr in [4/5, 4/6, 4/7, 4/8]:
            for cf in cf_candidates:
                N = len(signal)
                M = max(1, int(cf * N))
                Phi = generate_sensing_matrix(M, N, seed=random.randint(0,1000))
                y = compress(signal, Phi)
                sigma = np.std(y) * 10 ** (-noise_snr_db / 20)
                y_noisy = y + np.random.randn(M) * sigma
                x_hat = recover_bp(y_noisy, Phi, epsilon=0.1*sigma)
                q = quality_metric(signal, x_hat)
                if q >= q_min:
                    e = energy_tx(sf, cr, int(np.ceil(M/8)))
                    if e < best_e:
                        best_e = e
                        best_params = (sf, cr, cf)
    return best_params, best_e