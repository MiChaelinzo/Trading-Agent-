import os
import math
import logging

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  NIGHT CITY TRADING PROTOCOL - NEURAL OPS
#  "The Net is vast and infinite." - Ghost in the Shell
# ═══════════════════════════════════════════════════════════════════════════════


def sigmoid(x):
    """Sigmoid Activation — Neural signal normalization
    
    Compresses values to [0,1] range — keeping the daemon stable, choom.
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("💀 FLATLINE in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    
    Encodes market observation window into neural-compatible format.
    Like a braindance scroll of recent market moves.
    """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
