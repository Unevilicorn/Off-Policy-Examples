from dataclasses import dataclass

from SAC.sac_game_wrapper import  gym_halfcheetah, gym_pendulum, gym_swimmer

@dataclass
class SacEnvConfigs:
    env_name: str
    env_creator: callable
    num_episodes: int
    num_steps: int
    gamma: float
    tau: float
    batch_size: int
    memory_capacity: int
    learning_rate: float
    replay_type: str
    hidden_layers: list[int]

env_to_configs = {
    "gym-pendulum": SacEnvConfigs(
        env_name="Pendulum-v1",
        env_creator=gym_pendulum,
        num_episodes=300,
        num_steps=200,
        gamma=0.99,
        tau=0.01,
        batch_size=128,
        memory_capacity=100000,
        learning_rate=1e-5,
        replay_type="default",
        hidden_layers=[64, 64]
    ),
    "gym-swimmer": SacEnvConfigs(
        env_name="Swimmer-v4",
        env_creator=gym_swimmer,
        num_episodes=300,
        num_steps=1000,
        gamma=0.99,
        tau=0.01,
        batch_size=128,
        memory_capacity=100000,
        learning_rate=1e-5,
        replay_type="default",
        hidden_layers=[64, 64]
    ),
    "gym-halfcheetah": SacEnvConfigs(
        env_name="HalfCheetah-v4",
        env_creator=gym_halfcheetah,
        num_episodes=300,
        num_steps=1000,
        gamma=0.99,
        tau=0.01,
        batch_size=128,
        memory_capacity=100000,
        learning_rate=1e-5,
        replay_type="default",
        hidden_layers=[128, 128]    
    )


}