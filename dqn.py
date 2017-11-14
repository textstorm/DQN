
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque

class LinearSchedule(object):
  def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
    self.schedule_timesteps = schedule_timesteps
    self.final_p = final_p
    self.initial_p = initial_p

  def value(self, t):
    interval = min(float(t) / self.schedule_timesteps, 1.0)
    return self.initial_p - interval * (self.initial_p - self.final_p)

class DeepQ(object):
  def __init__(self, args, env, name):
    self.observation_space_shape = env.observation_space.shape[0]
    self.num_actions = env.action_space.n
    self.learning_rate = args.learning_rate
    self.replay_memory = deque()

    self.memory_size = args.memory_size
    self.epsilon = args.start_epsilon
    self.start_epsilon = args.start_epsilon
    self.final_epsilon = args.final_epsilon
    self.batch_size = args.batch_size
    self.gamma = args.gamma
    self.max_grad_norm = args.max_grad_norm

    self.time_step = 0
    self.schedule_timesteps = args.schedule_timesteps
    self.exploration = LinearSchedule(self.schedule_timesteps, 0.1)

    self.sess = tf.InteractiveSession()

    self.add_placeholder()
    self.build_Q()
    self.build_train()

    self.sess.run(tf.global_variables_initializer())

  def add_placeholder(self):
    self.state_t = tf.placeholder(tf.float32, 
                                  [None, self.observation_space_shape], 
                                  name="state_t")
    self.action_t = tf.placeholder(tf.float32,
                                   [None, self.num_actions],
                                   name="action_t")
    self.y_input = tf.placeholder(tf.float32,
                                  [None],
                                  name='y_t')

  def build_Q(self):
    W1 = self._weight_variable([self.observation_space_shape, 20], 'W1')
    b1 = self._bias_variable([20], 'b1')
    W2 = self._weight_variable([20, self.num_actions], 'W2')
    b2 = self._bias_variable([self.num_actions], 'b2')
    h_layer = tf.nn.relu(tf.matmul(self.state_t, W1) + b1)

    self.Q_value = tf.matmul(h_layer, W2) + b2

  def build_train(self):
    action_Q = tf.reduce_sum(self.Q_value * self.action_t, 1)
    self.loss_op = tf.reduce_mean(tf.square(self.y_input - action_Q))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)

  def egreedy_policy(self, state):
    self.epsilon = self.exploration.value(self.time_step)
    Q_value = self.Q_value.eval(
      feed_dict={self.state_t: [state]})[0]

    random_action = random.randint(0, self.num_actions - 1)
    deterministic_action = np.argmax(Q_value)

    if random.random() <= self.epsilon:
      action = random_action
      # self.epsilon -= (self.start_epsilon - self.final_epsilon) / 10000
      # if self.epsilon <= self.final_epsilon:
      #   self.epsilon = self.final_epsilon
    else:
      action = deterministic_action
    self.time_step += 1
    return action

  def train(self, state, action, reward, next_state, done):
    one_hot_action = np.zeros(self.num_actions)
    one_hot_action[action] = 1
    self.replay_memory.append((state, one_hot_action, reward, next_state, done))
    if len(self.replay_memory) > self.memory_size:
      self.replay_memory.popleft()

    if len(self.replay_memory) > self.batch_size:
      self.train_Q()

  def train_Q(self):
    self.time_step += 1
    batch_data = random.sample(self.replay_memory, self.batch_size)
    state_batch = [data[0] for data in batch_data]
    action_batch = [data[1] for data in batch_data]
    reward_batch = [data[2] for data in batch_data]
    next_state_batch = [data[3] for data in batch_data]

    y_batch = []
    Q_left_batch = self.Q_value.eval(feed_dict={self.state_t: next_state_batch})

    for i in range(0, self.batch_size):
      done = batch_data[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + self.gamma * np.max(Q_left_batch[i]))

    self.optimizer.run(
      feed_dict={self.y_input: y_batch,
                 self.action_t: action_batch,
                 self.state_t: state_batch})

  def action(self, state):
    return np.argmax(self.Q_value.eval(
      feed_dict = {self.state_t:[state]})[0])

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
