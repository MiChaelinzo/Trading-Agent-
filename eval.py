"""
═══════════════════════════════════════════════════════════════════════════════
  NETRUNNER TRADING DAEMON - EVALUATION PROTOCOL v2.077
  "In Night City, you either extract eddies or get flatlined." - V
═══════════════════════════════════════════════════════════════════════════════

Script for evaluating the NetRunner Trading Daemon on live market data.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day market observation window. [default: 10]
  --model-name=<model-name>     Name of the neural shard to deploy (evaluates all in `models/` if unspecified).
  --debug                       Enable verbose NetRunner logging.
"""

import os
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)


def main(eval_stock, window_size, model_name, debug):
    """ 
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  DAEMON DEPLOYMENT PROTOCOL                                           ║
    ║  Unleashing the trading ICE-breaker on corpo markets...               ║
    ║  Reference: https://arxiv.org/abs/1312.5602                           ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """    
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    # Single Model Evaluation — Deploy specific neural shard
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)
        
    # Multiple Model Evaluation — Test all available chrome
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained=True, model_name=model)
                profit = evaluate_model(agent, data, window_size, debug)
                show_eval_result(model, profit, initial_offset)
                del agent


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["<eval-stock>"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
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
    ║               TRADING DAEMON v2.077 - EVALUATION MODE                  ║
    ║                   "Time to see what this chrome can do."               ║
    ║                                                                        ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║   > NEURAL LINK: ESTABLISHED                                           ║
    ║   > MARKET DATA: LOADED                                                ║
    ║   > STATUS: DEPLOYING DAEMON...                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        main(eval_stock, window_size, model_name, debug)
    except KeyboardInterrupt:
        print("\n💀 FLATLINE: Evaluation aborted by user. Stay safe out there, choom.")
