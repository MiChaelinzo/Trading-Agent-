import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


# ═══════════════════════════════════════════════════════════════════════════════
#  NIGHT CITY TRADING PROTOCOL - CORE METHODS
#  "Money talks. In Night City, it screams." - Unknown Fixer
# ═══════════════════════════════════════════════════════════════════════════════


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    """Neural Training Loop — Teaching the daemon to extract eddies
    """
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, 
                  desc='⚡ NEURAL CYCLE {}/{}'.format(episode, ep_count)):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # Select combat action
        action = agent.act(state)

        # 🟢 BUY — Acquiring corpo assets
        if action == 1:
            agent.inventory.append(data[t])

        # 🔴 SELL — Extracting eddies
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta

        # ⚪ HOLD — Maintaining position
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # Save neural shard every 10 cycles
    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    """Evaluation Protocol — Daemon deployed on live market data
    
    Time to see if this chrome was worth the eddies, choom.
    """
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    
    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # Select combat action (evaluation mode - no chaos)
        action = agent.act(state, is_eval=True)

        # 🟢 BUY — Acquiring corpo assets
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("🟢 ACQUIRING: {}".format(format_currency(data[t])))
        
        # 🔴 SELL — Extracting eddies from the market
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("🔴 EXTRACTING: {} | Eddies: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        
        # ⚪ HOLD — Ghost in the machine
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
