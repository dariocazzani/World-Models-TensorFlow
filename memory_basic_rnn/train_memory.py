import numpy as np
import glob
import argparse

from model import Model
import tensorflow as tf

_SEQ_MAX_LENGTH = 300

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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--rnn_size', type=int, default=256,
						help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
						help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
						help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=64,
						help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=_SEQ_MAX_LENGTH,
						help='RNN sequence length')
	parser.add_argument('--num_epochs', type=int, default=30,
						help='number of epochs')
	parser.add_argument('--keep_prob', type=float, default=0.8,
						help='dropout keep probability')
	parser.add_argument('--grad_clip', type=float, default=10.,
						help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.005,
						help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.95,
						help='decay rate for rmsprop')
	args = parser.parse_args()

	data_gen = data_iterator(args.batch_size)
	data, targets = next(data_gen)
	print(data.shape)
	print(targets.shape)

	model = Model(args)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for e in range(args.num_epochs):
			sess.run(tf.assign(model.lr,
			args.learning_rate * (args.decay_rate ** e)))
			state = model.state_in.eval()
			for b in range(int(1000/args.batch_size)):
				ith_train_step = e * int(1000/args.batch_size) + b
				data, targets = next(data_gen)
				feed = {
					model.input_data: data,
					model.target_data: targets,
					model.state_in: state}
				train_loss, state, _ = sess.run(
					[model.cost, model.state_out, model.train_op], feed)

				print("(epoch {}), train_loss = {:.3f}".format(e, train_loss))

if __name__ == '__main__':
	main()
