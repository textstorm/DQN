
import config
import dqn
import gym
import numpy as np
import tensorflow as tf

def main(args):
  env = gym.make(args.env_name)
  args.observation_space_shape = env.observation_space.shape[0]
  args.action_space_n = env.action_space.n

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = dqn.DeepQ(args, sess, name='dqn')
    for episode in range(args.nb_episodes):
      obs = env.reset()
      for step in range(args.nb_steps):
        action = model.egreedy_action(np.array(obs)[None])[0]
        new_obs, reward, done, _ = env.step(action)
        if done: reward = 0.
        model.train(obs, action, reward, new_obs, done)
        obs = new_obs
        if done: break

      if episode % 1000 == 0:
        total_reward = 0
        for i in range(args.test_episodes):
          obs = env.reset()
          for j in range(args.nb_steps):
            env.render()
            action = model.action(state)
            state,reward,done,_ = env.step(action)
            total_reward += reward
            if done: break
        ave_reward = total_reward / args.test_episodes
        print 'episode: ',episode,'Evaluation Average Reward:',ave_reward

if __name__ == '__main__':
  args = config.get_args()
  main(args)