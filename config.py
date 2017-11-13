
import argparse

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default='1013', help='random seed')
  parser.add_argument('--env_name', type=str, default='CartPole-v0', help='name of environment')

  parser.add_argument('--hiddens_size', type=int, default=[64], help='mlp hidden size')
  parser.add_argument('--memory_size', type=int, default=50000, help="experience replay memory size")

  parser.add_argument('--nb_episodes', type=int, default=1, help='number of episode')
  parser.add_argument('--test_episodes', type=int, default=5, help='number of test episode')
  parser.add_argument('--nb_steps', type=int, default=1000, help='number of step in one episode')
  parser.add_argument('--start_epsilon', type=float, default=1.0, help='')
  parser.add_argument('--final_epsilon', type=float, default=0.05, help='')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--gamma', type=float, default=0.9, help='discount foctor')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max norm of gradient')

  return parser.parse_args()
