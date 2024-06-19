# Off Policy Algorithms
This repository contains the implementation of off policy algorithms. As well as implementations from TianShou and StableBaseline3.

The algorithms implemented are:
- Deep Q Learning (DQN)
- Soft Actor Critic (SAC)

## How to run the code
- First install the requirements by running `pip install -r requirements.txt`
- Then run the code by running `python dqn.py` or `python sac.py`
- If you want to run the code with the stablebaseline3 implementation, run `python dqn_stable.py` or `python sac_stable.py`
- If you want to run the code with the tianshou implementation, run `python dqn_tianshou.py` or `python sac_tianshou.py`
- By default it will run the pendulum environment, if you want to run the code with a different environment, you can change add it to the command line arguments. For example `python dqn.py gym-swimmer`, currently the accepted environments are `gym-pendulum`, `gym-swimmer`, `gym-reacher` and `gym-halfcheetah`
