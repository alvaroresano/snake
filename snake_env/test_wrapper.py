# test_wrapper.py

import gymnasium as gym
import snake_env # Para registrar los entornos base

# Importa tu clase Wrapper directamente en este script
from snake_env.snake_wrapper import NoGrowthRewardWrapper

# 1. Crea el entorno base que registraste sin el wrapper
base_env = gym.make("Snake-Base-For-Wrapping-v0")  # , render_mode="human"  para ver

# 2. "Envuelve" el entorno base con tu clase wrapper
env = NoGrowthRewardWrapper(base_env)


obs, _ = env.reset()
done = False
steps = 0
total_reward = 0

print("--- Iniciando prueba con el entorno envuelto manualmente ---")

while not done:
    steps += 1
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    
    if reward != 0:
        print(f'Paso: {steps} | Acción: {action} | Recompensa: {reward:.2f} | Recompensa Total: {total_reward:.2f}')

    if terminated or truncated:
        done = True

print(f"\n--- Episodio finalizado en {steps} pasos ---")
print(f"Recompensa final total: {total_reward:.2f}")
    
env.close()