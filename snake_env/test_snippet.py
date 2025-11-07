import gymnasium as gym
import snake_env  # Este importa el __init__.py y registra el entorno

env = gym.make("Snake-v0", render_mode="human")
#env = gym.make("Snake-v0-step5", render_mode="human")

obs, _ = env.reset()
done = False
steps = 0

while not done:
    steps += 1
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    #print(f'{steps} - Obs:{obs}')
    #if steps > 10: break
    
env.close()