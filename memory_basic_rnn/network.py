import tensorflow as tf
import numpy as np

_EMBEDDING_SIZE = 32
_ACTION_SIZE = 2

class Network():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        cell_fn = tf.contrib.rnn.GRUCell
        self.state_zero = np.zeros((1, args.rnn_size))
        # if args.model == 'rnn':
        #     cell_fn = tf.contrib.rnn.BasicRNNCell
        #     self.state_zero = np.zeros((1, args.rnn_size))
        # elif args.model == 'gru':
        #
        #     cell_fn = tf.contrib.rnn.GRUCell
        #     self.state_zero = np.zeros((1, args.rnn_size))
        # elif args.model == 'lstm':
        #     cell_fn = tf.contrib.rnn.BasicLSTMCell
        #     self.state_zero = np.zeros((1, args.rnn_size*2))
        # else:
        #     raise Exception("model type not supported: {}".format(args.model))

        def get_cell():
            return cell_fn(args.rnn_size)#, state_is_tuple=False)

        cell = tf.contrib.rnn.MultiRNNCell(
            [get_cell() for _ in range(args.num_layers)])

        if (infer == False and args.keep_prob < 1):  # training mode
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, args.seq_length, _EMBEDDING_SIZE + _ACTION_SIZE],
                name='data_in')
        self.target_data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, args.seq_length, _EMBEDDING_SIZE], name='targets')

        # zero_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        # self.state_in = tf.identity(zero_state, name='state_in')
        self.state_in = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # with tf.variable_scope('rnnlm'):
        #     output_w = tf.get_variable("output_w", [args.rnn_size, _EMBEDDING_SIZE])
        #     output_b = tf.get_variable("output_b", [_EMBEDDING_SIZE])

        inputs = tf.unstack(self.input_data, axis=1)

        outputs, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(
            inputs, self.state_in, cell, loop_function=None)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
        self.output = tf.layers.dense(output, units=_EMBEDDING_SIZE, activation=None)

        self.state_out = tf.identity(state_out, name='state_out')

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_data, [-1, _EMBEDDING_SIZE])

        self.cost = tf.losses.mean_squared_error(flat_target_data, self.output)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, prev_obs, prev_state):
        feed = {self.input_data: prev_obs, self.state_in: prev_state}
        next_obs, next_state = sess.run([self.output, self.state_out], feed)
        return next_obs, next_state

    def sample_first(self, sess, prev_obs):
        # prev_obs = np.zeros((1, 1, _EMBEDDING_SIZE + _ACTION_SIZE), dtype=np.float32)
        return self.sample(sess, prev_obs, self.state_zero)
