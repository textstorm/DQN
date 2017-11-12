
import argsparse

def get_args():

  parser = argsparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default='1013', help='random seed')

  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--hiddens_size', type=int, default=[64], help='mlp hidden size')
  parser.add_argument('--memory_size', type=int, default=50000, helo="experience replay memory size")

  parser.add_argument('--max_step', type=int, default=1, help='number of epoch')
  parser.add_argument('--epsilon', type=float, default=1.0, help='')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max norm of gradient')

  return parser.parse_args()
