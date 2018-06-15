import os
import gym
import numpy as np
import wrapper
import argparse
import time
NAME = 'lander-rt'
GENERATIONS = 10


class LanderRtNeat(wrapper.RtNeatWrapper):
    def play_winner(self, winner):
        env = gym.make('LunarLanderContinuous-v2')

        winner_net = self.create_net(winner)

        ob = env.reset()
        while True:
            action = winner_net.activate(ob)
            ob, reward, done, info = env.step(np.array(action))
            env.render()
            if done:
                ob = env.reset()


def main(args):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '{}.ini'.format(NAME))

    if args.checkpoint:
        w = LanderRtNeat(NAME, config_path, checkpoint=args.checkpoint, checkpoint_interval=args.save_interval)
    else:
        w = LanderRtNeat(NAME, config_path)

    if args.command == "train":

        env = gym.make('LunarLanderContinuous-v2')
        while True:
            for genome_id in w.rt_get_population_ids():
                ob = env.reset()
                reward_total = 0
                while True:
                    action = w.rt_activate(genome_id, ob)
                    if action is None:
                        break
                    ob, reward, done, info = env.step(np.array(action))
                    reward_total = reward_total + reward
                    if done:
                        w.rt_set_fitness(genome_id, reward_total)
                        break

            w.rt_iterate()



    elif args.command == "play":
        w.play()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--checkpoint')
    parser.add_argument('--generations', default=300, type=int)
    parser.add_argument('--save_interval', default=20)
    main(parser.parse_args())
