
import config
import dqn
import gym
import numpy as np
import tensorflow as tf

def main(args):
  env = gym.make(args.env_name)
  model = dqn.DeepQ(args, env, name='dqn')

  for episode in range(args.nb_episodes):
    state = env.reset()
    for step in range(args.nb_steps):
      #action = model.egreedy_action(np.array(obs)[None])[0]
      action = model.egreedy_policy(state)
      next_state, reward, done, _ = env.step(action)
      model.train(state, action, reward, next_state, done)
      state = next_state
      if done: break

    if episode % 10 == 0:
      print model.epsilon
      print model.replay_memory.get_length()
      total_reward = 0
      for i in range(args.test_episodes):
        state = env.reset()
        for j in range(args.nb_steps):
          #env.render()
          action = model.action(state)
          state, reward, done, _ = env.step(action)
          total_reward += reward
          if done: 
            break

      ave_reward = total_reward / args.test_episodes
      print "Episode: %d, Evaluation Average Reward: %.1f" % (episode, ave_reward)
      if ave_reward >= 200:
        break

if __name__ == '__main__':
  args = config.get_args()
  main(args)