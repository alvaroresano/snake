import gymnasium as gym
import numpy as np

# Estas funciones auxiliares son necesarias para que el wrapper determine si hubo una colisión.
# Se basan en el 'env.py' que proporcionaste, donde 'tableSize' es 500.
def collision_with_boundaries(snake_head):
    tableSize = 500 
    if snake_head[0] >= tableSize or snake_head[0] < 0 or snake_head[1] >= tableSize or snake_head[1] < 0:
        return True
    return False

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    return False


class NoGrowthRewardWrapper(gym.Wrapper):
    """
    Wrapper para el entorno Snake que implementa una lógica de recompensa compleja:
    - La serpiente no crece al comer una manzana.
    - Recompensa de +1000 al comer.
    - Penalización de -10000 al chocar.
    - +5 si se acerca a la manzana, -1 si se aleja.
    - Penalización acumulativa por cada paso sin comer.
    - Penalización constante basada en la distancia a la manzana.
    """
    def __init__(self, env):
        super(NoGrowthRewardWrapper, self).__init__(env)
        # Inicializamos las variables de estado del wrapper
        self.steps_since_last_apple = 0
        self.previous_distance_to_apple = 0

    def reset(self, **kwargs):
        # Reseteamos el entorno base y obtenemos la observación inicial
        observation, info = self.env.reset(**kwargs)
        
        # Reseteamos las variables de estado del wrapper
        self.steps_since_last_apple = 0
        
        # Calculamos y guardamos la distancia inicial para la primera comparación en step()
        snake_head = np.array(self.unwrapped.snake_head)
        apple_position = np.array(self.unwrapped.apple_position)
        self.previous_distance_to_apple = np.linalg.norm(snake_head - apple_position)
        
        return observation, info

    def step(self, action):
        # 1. Ejecutamos un paso en el entorno original
        observation, reward, terminated, truncated, info = self.env.step(action)

        # 2. Obtenemos el estado actual del juego usando .unwrapped para saltar el TimeLimit wrapper
        snake_head = np.array(self.unwrapped.snake_head)
        apple_position = np.array(self.unwrapped.apple_position)
        
        # 3. Comprobamos las condiciones del juego
        ate_apple = np.array_equal(snake_head, apple_position)
        # Usamos las funciones auxiliares para determinar si hubo colisión
        is_collision = collision_with_boundaries(snake_head) or collision_with_self(self.unwrapped.snake_position)

        # 4. Implementamos la lógica de "no crecimiento"
        if ate_apple:
            self.unwrapped.snake_position.pop()

        # 5. Calculamos la recompensa según tus reglas
        if is_collision:
            # Recompensa por chocar (termina el episodio)
            reward = -10000
        elif ate_apple:
            # Recompensa por comer la manzana
            reward = 1000
            # Reseteamos el contador de pasos
            self.steps_since_last_apple = 0
        else:
            # Si es un movimiento normal, construimos la recompensa por partes
            reward = 0
            
            # Recompensa/Penalización por acercarse/alejarse
            current_distance = np.linalg.norm(snake_head - apple_position)
            if current_distance < self.previous_distance_to_apple:
                reward += 5  # +5 por acercarse
            else:
                reward -= 1  # -1 por alejarse
            
            # Actualizamos la distancia para el próximo paso
            self.previous_distance_to_apple = current_distance
            
            # Penalización por tiempo (aumenta con cada paso)
            self.steps_since_last_apple += 1
            reward -= 0.01 * self.steps_since_last_apple
            
            # Penalización constante por distancia
            reward -= current_distance / 100

        return observation, reward, terminated, truncated, info