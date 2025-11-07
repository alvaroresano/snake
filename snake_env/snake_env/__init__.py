from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
)

register(
    id="Snake-v0-step5",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    kwargs={"action_size":2},
)

register(
    id="Snake-v1",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=100,
)

register(
    id="Snake-v2",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    reward_threshold=30,
)