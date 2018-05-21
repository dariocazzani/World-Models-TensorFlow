import numpy as np
import glob

def data_iterator(batch_size):
	obs_data_files = glob.glob('../data/obs_data_*')
	act_data_files = glob.glob('../data/act_data_*')
	while True:
		obs = np.load(random.sample(obs_data_files, 1)[0])
		act = np.load(random.sample(act_data_files, 1)[0])
		N = data.shape[0]
		start = np.random.randint(0, N-batch_size-1)
        current_obs = obs[start:start+batch_size]
        current_act = act[start:start+batch_size]
        future_obs = obs[start+1:start+1+batch_size]
        yield current_obs, current_act, future_obs
