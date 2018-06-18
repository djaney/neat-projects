import os
import gym
import numpy as np
import wrapper
import argparse

NAME = 'pole'
GENERATIONS = 10


class PoleNeat(wrapper.NeatWrapper):
    def eval_genomes(self, genomes, config):
        env = gym.make('CartPole-v1')

        for genome_id, genome in genomes:
            net = self.create_net(genome)

            ob = env.reset()
            reward_sum = 0
            while True:
                action = net.activate(ob)
                ob, reward, done, info = env.step(np.argmax(action))
                reward_sum += reward
                if done:
                    break
            genome.fitness = reward_sum

    def play_winner(self, winner):
        env = gym.make('CartPole-v1')

        winner_net = self.create_net(winner)

        ob = env.reset()
        while True:
            action = winner_net.activate(ob)
            ob, reward, done, info = env.step(np.argmax(action))
            env.render()
            if done:
                ob = env.reset()


def main(args):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '{}.ini'.format(NAME))

    if args.checkpoint:
        w = PoleNeat(NAME, config_path, checkpoint=args.checkpoint, checkpoint_interval=args.save_interval)
    else:
        w = PoleNeat(NAME, config_path)

    if args.command == "train":
        w.serve()
    elif args.command == "play":
        w.play()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--checkpoint')
    parser.add_argument('--generations', default=300, type=int)
    parser.add_argument('--save_interval', default=20)
    main(parser.parse_args())
