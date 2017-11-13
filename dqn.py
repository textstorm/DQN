
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque

class DeepQ():
  def __init__(self, args, session, name):
    self.observation_space_shape = args.observation_space_shape
    self.num_actions = args.action_space_n
    self.learning_rate = args.learning_rate
    self.sess = session
    self.replay_memory = deque()

    self.hiddens = args.hiddens_size
    self.memory_size = args.memory_size
    self.epsilon = args.start_epsilon
    self.start_epsilon = args.start_epsilon
    self.final_epsilon = args.final_epsilon
    self.batch_size = args.batch_size
    self.gamma = args.gamma
    self.max_grad_norm = args.max_grad_norm

    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    # self.global_step = tf.get_variable(
    #   'global_step', [], 'int32', tf.constant_initializer(0), trainable=False)

    self.add_placeholder()
    self.build_graph()
    self.build_train()

  def add_placeholder(self):
    self.observ_t = tf.placeholder(tf.float32, 
                                   [None, self.observation_space_shape], 
                                   name="observe_t")
    self.action_t = tf.placeholder(tf.float32,
                                   [None, self.num_actions],
                                   name="action_t")
    # self.reward_t = tf.placeholder(tf.float32,
    #                                [None],
    #                                name='reward_t')
    self.y_t = tf.placeholder(tf.float32,
                              [None, self.num_actions],
                              name='y_t')

  def build_graph(self):
    self.q_values = self.mlp()
    # action_t = self.egreedy_action(q_values)
    # new_ob, reward, done, _ = self.env.step(action_t)
    # if done: reward = 0.
    # self.replay_memory.append((new_ob, reward, done))


  def mlp(self):
    W1 = self._weight_variable([self.observation_space_shape, 20], 'W1')
    b1 = self._bias_variable([20], 'b1')
    W2 = self._weight_variable([20, self.num_actions], 'W2')
    b2 = self._bias_variable([self.num_actions], 'b2')
    h_layer = tf.nn.relu(tf.matmul(self.observ_t, W1) + b1)
    return tf.matmul(h_layer, W2) + b2

  def egreedy_action(self, observ_t):
    random_action = tf.random_uniform(
      [1], maxval=self.num_actions, minval=0, dtype=tf.int32)
    deterministic_action = tf.argmax(self.q_values, 1)
    choose_action = tf.random_uniform(
      [1], maxval=1, minval=0, dtype=tf.float32) < self.epsilon

    self.epsilon -= (self.start_epsilon - self.final_epsilon) / 10000
    if self.epsilon <= self.final_epsilon:
      self.epsilon = self.final_epsilon
    action = tf.where(choose_action, random_action, deterministic_action)
    return self.sess.run(action, feed_dict={self.observ_t: observ_t})

  # def egreedy_action(self, q_values):
  #   random_actions = tf.random_uniform(tf.stack([self.batch_size],)
  #                                      maxval=self.num_actions,
  #                                      minval=0,
  #                                      dtype=tf.int32)
  #   deterministic_actions = tf.argmax(q_values, 1)
  #   choose_action = tf.random_uniform(
  #     tf.stack(self.batch_size), maxval=1.0, minval=0.0, dtype=tf.float32) < self.epsilon
  #   return tf.where(choose_action, random_actions, deterministic_actions)

  def train(self, obs, action, reward, new_obs, done):
    one_hot_action = tf.one_hot(action, self.num_actions)
    self.replay_memory.append((obs, action, reward, new_obs, done))
    if len(self.replay_memory) > self.memory_size:
      self.replay_memory.popleft()
    if len(replay_memory) > self.batch_size:
      self.train_Q()

  def train_Q(self):
    batch_data = random.sample(self.replay_memory, self.batch_size)
    obs_batch = [data[0] for data in batch_data]
    act_batch = [data[1] for data in batch_data]
    rew_batch = [data[2] for data in batch_data]
    nobs_batch = [data[3] for data in batch_data]
    done_batch = [data[4] for data in batch_data]

    y_batch = []
    Q_value_batch = self.sess.run(self.Q_value, feed_dict={self.observ_t: nobs_batch})

    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(rew_batch[i])
      else :
        y_batch.append(rew_batch[i] + self.gamma * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_t:y_batch,
      self.action_t:action_batch,
      self.observ_t:state_batch})

  def build_train(self):
    q_value = tf.reduce_sum(self.q_values * self.action_t, 1)
    self.loss_op = tf.reduce_mean(tf.square(self.y_t - q_value))
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss_op)

  def action(self,state):
    return np.argmax(self.sess.run(
      self.Q_value, feed_dict = {self.state_input:[state]})[0])

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