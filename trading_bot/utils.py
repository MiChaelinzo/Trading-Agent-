import os
import math
import logging

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  NIGHT CITY TRADING PROTOCOL - UTILITY FUNCTIONS
#  "The street finds its own uses for things." - William Gibson
# ═══════════════════════════════════════════════════════════════════════════════

# Formats Position (Eddies gained/lost)
format_position = lambda price: ('💀 -€$' if price < 0 else '💰 +€$') + '{0:.2f}'.format(abs(price))


# Formats Currency (Eurodollars)
format_currency = lambda price: '€${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results - Night City style
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('⚡ NEURAL CYCLE {}/{} | Train Eddies: {} | Val Status: 💀 FLATLINED | Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('⚡ NEURAL CYCLE {}/{} | Train Eddies: {} | Val Eddies: {} | Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results - NetRunner output
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('🔴 DAEMON [{}]: 💀 FLATLINED - No eddies extracted\n'.format(model_name))
    else:
        logging.info('🟢 DAEMON [{}]: {} extracted from corpo markets\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file - Extracting corpo intel
    """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.
    Night City Protocol: CPU runs cooler for sequential ops, choom.
    """
    # In TensorFlow 2.x, we control GPU usage differently
    import tensorflow as tf
    # Disable GPU to use CPU for faster training
    try:
        # For TensorFlow 2.x
        tf.config.set_visible_devices([], 'GPU')
        logging.debug("⚡ CYBERWARE: GPU disabled, routing through CPU cortex")
    except Exception as e:
        # Fallback for older versions or environments without GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.debug("⚡ CYBERWARE: CUDA_VISIBLE_DEVICES=-1, CPU mode engaged")
