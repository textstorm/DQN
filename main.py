
import config
import dqn

def main(args):
  env = gym.make()

  with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    model = dqn.DeepQ(args, env, sess, name='dqn')
    for idx in range(args.max_step):
      

if __name__ == '__main__':
  args = config.get_args()
  main(args)