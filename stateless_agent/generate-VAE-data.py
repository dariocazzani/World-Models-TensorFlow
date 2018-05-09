import numpy as np
import multiprocessing as mp
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

_BATCH_SIZE = 16
_NUM_BATCHES = 16
_TIME_STEPS = 150
_RENDER = True

def generate_action(prev_action):
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action*mask

def normalize_observation(observation):
	return observation.astype('float32') / 255.

def simulate_batch(batch_num):
    env = CarRacing()

    obs_data = []
    action_data = []
    action = env.action_space.sample()
    for i_episode in range(_BATCH_SIZE):
        observation = env.reset()
        # Little hack to make the Car start at random positions in the race-track
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        observation = normalize_observation(observation)

        obs_sequence = []

        for _ in range(_TIME_STEPS):
            if _RENDER:
                env.render()

            action = generate_action(action)

            observation, reward, done, info = env.step(action)
            observation = normalize_observation(observation)

            obs_data.append(observation)

    print("Saving dataset for batch {}".format(batch_num))
    np.save('../data/obs_data_VAE_{}'.format(batch_num), obs_data)
    
    env.close()

def main():
    print("Generating data for env CarRacing-v0")

    with mp.Pool(mp.cpu_count()) as p:
        p.map(simulate_batch, range(_NUM_BATCHES))

if __name__ == "__main__":
    main()
