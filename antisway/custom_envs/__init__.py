from gymnasium.envs.registration import register

register(
    id="custom_envs/CartPendulum-v0",
    entry_point="custom_envs.envs.cartpendulum:CartPendulumEnv",
    max_episode_steps=200,
    reward_threshold=500,
)


'''register(
    id='custom_envs/CustomCartPole-v0',
    entry_point='custom_envs.envs:CustomCartPoleEnv',
    max_episode_steps=100,
    reward_threshold=500,
)'''

