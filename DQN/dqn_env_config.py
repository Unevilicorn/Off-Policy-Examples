from dqn_game_wrapper import discrete_gym_pendulum, discrete_gym_swimmer, discrete_gym_cheetah

from dataclasses import dataclass

@dataclass
class DqnEnvConfigs:
    env_name: str
    env_creator: callable
    num_episodes: int
    gamma: float
    max_steps: int
    batch_size: int
    epsilon_init: float
    epsilon_min: float
    epsilon_frac: float
    tau: float
    target_update: int
    memory_capacity: int
    learning_rate: float
    replay_type: str
    gradient_clip: float | None
    hidden_layers: list[int]
    action_space: int

env_to_configs = {
    "gym-pendulum": DqnEnvConfigs(
        env_name="Pendulum-v1",
        env_creator=discrete_gym_pendulum,
        num_episodes=300,
        gamma=0.99,
        max_steps=200,
        batch_size=128,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_frac=0.5,
        tau=0.01,
        target_update=1,
        memory_capacity=100000,
        learning_rate=1e-4,
        replay_type="default",
        gradient_clip=None,
        hidden_layers=[64, 64],
        action_space=9**1,
    ),
    "gym-swimmer": DqnEnvConfigs(
        env_name="Swimmer-v4",
        env_creator=discrete_gym_swimmer,
        num_episodes=300,
        gamma=0.99,
        max_steps=1000,
        batch_size=128,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_frac=0.5,
        tau=0.01,
        target_update=1,
        memory_capacity=100000,
        learning_rate=1e-5,
        replay_type="default",
        gradient_clip=None,
        hidden_layers=[64, 64],
        action_space=9**2,
    ),
    "gym-halfcheetah": DqnEnvConfigs(
        env_name="HalfCheetah-v4",
        env_creator=discrete_gym_cheetah,
        num_episodes=300,
        gamma=0.99,
        max_steps=1000,
        batch_size=128,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_frac=0.5,
        tau=0.01,
        target_update=1,
        memory_capacity=100000,
        learning_rate=1e-5,
        replay_type="default",
        gradient_clip=None,
        hidden_layers=[128, 128],
        action_space=3**6,
    )
}