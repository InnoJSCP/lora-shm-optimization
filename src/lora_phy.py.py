import numpy as np

# LoRa constants (from SX1276 datasheet)
BW = 125e3          # bandwidth [Hz]
P_TX_DBM = 14       # transmission power [dBm]
P_TX_MW = 10 ** (P_TX_DBM / 10)  # [mW]
N_PREAMBLE = 8
CRC = 1
H = 0               # explicit header mode
DE = 0
CR_MAP = {4/5: 1, 4/6: 2, 4/7: 3, 4/8: 4}

def get_nsym(payload_bytes, sf, cr_index):
    """Number of payload symbols (LoRa spec)."""
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * CRC - 20 * H
    den = 4 * (sf - 2 * DE)
    return int(np.ceil(num / den)) if den > 0 else 0

def time_on_air(sf, cr, payload_bytes):
    """Time‑on‑Air in seconds."""
    cr_idx = CR_MAP[cr]
    ts = (2 ** sf) / BW
    tpreamble = (N_PREAMBLE + 4.25) * ts
    nsym = get_nsym(payload_bytes, sf, cr_idx)
    tpayload = (8 + max(nsym * (cr_idx + 4), 0)) * ts
    return tpreamble + tpayload

def energy_tx(sf, cr, payload_bytes):
    """Energy consumption for transmission [mJ]."""
    return P_TX_MW * time_on_air(sf, cr, payload_bytes)