"""
═══════════════════════════════════════════════════════════════════════════════
  NETRUNNER TRADING DAEMON - TRAINING PROTOCOL v2.077
  "Wake up, Samurai. We have markets to burn." - Johnny Silverhand
═══════════════════════════════════════════════════════════════════════════════

Script for training the NetRunner Trading Daemon using Deep Q-Learning.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Combat algorithm for neural training:
                                      `dqn` - Vanilla DQN (basic chrome)
                                      `t-dqn` - DQN with fixed target (stabilized)
                                      `double-dqn` - Dual-core processing [default: t-dqn]
  --window-size=<window-size>       Size of the n-day market observation window
                                    (neural input vector size). [default: 10]
  --batch-size=<batch-size>         Number of braindance samples per training batch. [default: 32]
  --episode-count=<episode-count>   Number of neural cycles for training. [default: 50]
  --model-name=<model-name>         Name of the neural shard to save/load. [default: model_debug]
  --pretrained                      Load existing neural shard and continue training.
  --debug                           Enable verbose NetRunner logging.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ 
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  NEURAL TRAINING PROTOCOL                                             ║
    ║  Jacking into the corpo markets...                                    ║
    ║  Reference: https://arxiv.org/abs/1312.5602                           ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║   ███╗   ██╗███████╗████████╗██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗    ║
    ║   ████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║   ██║████╗  ██║████╗  ██║    ║
    ║   ██╔██╗ ██║█████╗     ██║   ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║    ║
    ║   ██║╚██╗██║██╔══╝     ██║   ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║    ║
    ║   ██║ ╚████║███████╗   ██║   ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║    ║
    ║   ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝    ║
    ║                                                                        ║
    ║               TRADING DAEMON v2.077 - TRAINING MODE                    ║
    ║                    "Time to extract some eddies."                      ║
    ║                                                                        ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║   > NEURAL LINK: ESTABLISHED                                           ║
    ║   > COMBAT ALGORITHM: {:<42}   ║
    ║   > MARKET DATA: LOADED                                                ║
    ║   > STATUS: JACKING IN...                                              ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """.format(strategy.upper()))

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("\n💀 FLATLINE: Training aborted by user. Stay safe out there, choom.")
