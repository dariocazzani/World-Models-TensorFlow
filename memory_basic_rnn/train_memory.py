import numpy as np
import glob
import argparse

from network import Network
import tensorflow as tf

model_path = "saved_models/"
model_name = model_path + 'model'

_SEQ_MAX_LENGTH = 140

def data_iterator(batch_size):
	obs_data_files = glob.glob('../data/obs_encoded_*')
	act_data_files = glob.glob('../data/act_data_*')

	def get_sequence():
		rollout = np.random.randint(len(obs_data_files))
		obs = np.load(obs_data_files[rollout])
		act = np.load(act_data_files[rollout])
		N = obs.shape[0]

		start = np.random.randint(1, N-_SEQ_MAX_LENGTH)
		current_obs = obs[start-1:start+_SEQ_MAX_LENGTH-1]
		current_act = act[start:start+_SEQ_MAX_LENGTH]
		future_obs = obs[start:start+_SEQ_MAX_LENGTH]

		return np.concatenate((current_obs, current_act), axis=1), future_obs

	while True:
		data_batch = []
		target_batch = []
		for _ in range(batch_size):
			data, target = get_sequence()
			data_batch.append(data)
			target_batch.append(target)
		yield np.array(data_batch), np.array(target_batch)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--rnn_size', type=int, default=256,
						help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=1,
						help='number of layers in the RNN')
	# parser.add_argument('--model', type=str, default='lstm',
	# 					help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=64,
						help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=_SEQ_MAX_LENGTH,
						help='RNN sequence length')
	parser.add_argument('--keep_prob', type=float, default=0.8,
						help='dropout keep probability')
	parser.add_argument('--grad_clip', type=float, default=10.,
						help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=1E-3,
						help='learning rate')
	args = parser.parse_args()
	return args

def train_M():
	args = parse_args()

	data_gen = data_iterator(args.batch_size)
	data, targets = next(data_gen)

	sess = tf.InteractiveSession()
	global_step = tf.Variable(0, name='global_step', trainable=False)
	writer = tf.summary.FileWriter('logdir')
	network = Network(args)
	tf.global_variables_initializer().run()

	saver = tf.train.Saver(max_to_keep=1)
	step = global_step.eval()

	try:
		saver.restore(sess, tf.train.latest_checkpoint(model_path))
		print("Model restored from: {}".format(model_path))
	except:
		print("Could not restore saved model")

	try:
		while True:
			state = sess.run(network.state_in)
			data, targets = next(data_gen)
			feed_dict = {
				network.input_data: data,
				network.target_data: targets,
				network.state_in: state}
			train_loss, state, _ = sess.run(
				[network.cost, network.state_out, network.train_op], feed_dict=feed_dict)

			if step % 10 == 0 and step > 0:
				print("step {}:, train_loss = {:.3f}".format(step, train_loss))
				save_path = saver.save(sess, model_name, global_step=global_step)

			step+=1

	except (KeyboardInterrupt, SystemExit):
		print("Manual Interrupt")

	except Exception as e:
		print("Exception: {}".format(e))

def load_M(model_path=model_path):
	args = parse_args()

	graph_rnn = tf.Graph()
	with graph_rnn.as_default():

		network = Network(args, infer=True)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.InteractiveSession(config=config)
		tf.global_variables_initializer().run()
		# sess = tf.Session(config=config, graph=graph_rnn)

		saver = tf.train.Saver()

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
		except:
			raise ImportError("Could not restore saved model")

		return sess, network

if __name__ == '__main__':
	train_M()
