import numpy as np
import uuid
import multiprocessing as mp
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

_ROLLOUTS = 1000
_TIME_STEPS = 350
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

def simulate_rollout(rollout_num):
    env = CarRacing()
    action = env.action_space.sample()
    observation = env.reset()

    # Little hack to make the Car start at random positions in the race-track
    position = np.random.randint(len(env.track))
    env.car = Car(env.world, *env.track[position][1:4])
    observation = normalize_observation(observation)

    obs_data = []
    act_data = []
    for _ in range(_TIME_STEPS):
        if _RENDER:
            env.render()

        action = generate_action(action)

        observation, reward, done, info = env.step(action)
        observation = normalize_observation(observation)

        obs_data.append(observation)
        # Save gas and brake as 1 variable:
        act_2D = np.zeros(2)
        act_2D[0] = action[0]
        act_2D[1] = action[2] - action[1] # negative values of act_2D means --> gas
        act_data.append(act_2D)

    print("Saving rollout {}".format(rollout_num))
    np.save('data/obs_data_{}'.format(rollout_num), obs_data)
    np.save('data/act_data_{}'.format(rollout_num), act_data)

    env.close()

def main():
    print("Generating data for env CarRacing-v0")

    with mp.Pool(mp.cpu_count()*2) as p:
        p.map(simulate_rollout, range(_ROLLOUTS))

if __name__ == "__main__":
    main()
