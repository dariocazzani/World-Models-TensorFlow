import numpy as np
import glob, tqdm
from stateless_agent.train_VAE import load_vae

if __name__ == '__main__':
    sess, network = load_vae('stateless_agent/saved_models')
    obs_data_files = glob.glob('data/obs_data_*')

    for idx, data_file in enumerate(tqdm.tqdm(obs_data_files)):
        obs = np.load(data_file)
        embedding = sess.run(network.z, feed_dict={network.image: obs})
        np.save('data/obs_encoded_{}'.format(idx), embedding)
