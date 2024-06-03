import sys
import os
from datetime import datetime
from matplotlib import pyplot as plt

def dqn_cli(func, path_to_save):
    # get the first argument
    if len(sys.argv) < 2:
        env = "gym-pendulum"
        use_wandb = False
    else:
        env = sys.argv[1]
        use_wandb = sys.argv[2].lower() == "true"
    print(f"Running for {env}")
    print(f"Using wandb: {use_wandb}")
    
    path_to_save = os.path.join(path_to_save, env)
    func(env, path_to_save, use_wandb=False)

def plot_and_save_average_plots(rewards, save_path, show=False, window=5):
    # Save the plot in the save_path folder, if it does not exist, create it
    plt.plot(rewards)
    plt.plot([sum(rewards[i:i+window])/window for i in range(len(rewards)-window)])
    
    if show:
        plt.show()
    
    if save_path is None:
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".png"))
