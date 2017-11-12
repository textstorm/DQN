
import tensorflow as tf
import tensorflow.contrib.layers as layers

class DeepQ():
  def __init__(self, args, env, session, name):
    self.observation_space_shape = env.observation_space.shape
    self.num_action = env.action_space.n
    self.hiddens = args.hiddens_size
    self.epsilon = args.epsilon

  def add_placeholder(self):
    self.observ_t = tf.placeholder(tf.float32, 
                                   [None, self.observation_space_shape], 
                                   name="observe_t")
    self.action_t = tf.placeholder(tf.float32,
                                 [None, self.num_action],
                                 name="action_t")


  def build_graph(self):
    q_t = self.mlp()
    q_value_t = tf.reduce_sum(q_t * tf.one_hot(self.action_t, self.num_action), 1)


  def mlp(self):
    W1 = self._weight_variable([self.observation_space_shape, 20])
    b1 = self._bias_variable([20])
    W2 = self._weight_variable([20, self.num_action])
    b2 = self._bias_variable([self.num_action])
    h_layer = tf.nn.relu(tf.matmul(self.observ_t, W1) + b1)
    return tf.matmul(h_layer, W2) + b2

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)