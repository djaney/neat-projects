import os
import gym
import numpy as np
import wrapper

NAME = 'pole'
GENERATIONS = 10


class PoleNeat(wrapper.Neat):
    def eval_genomes(self, genomes, config):
        env = gym.make('LunarLanderContinuous-v2')

        for genome_id, genome in genomes:
            net = self.create_net(genome)

            ob = env.reset()
            reward_sum = 0
            while True:
                action = net.activate(ob)
                ob, reward, done, info = env.step(np.array(action))
                reward_sum = reward_sum + reward
                if done:
                    break
                genome.fitness = reward_sum

    def play(self):
        env = gym.make('LunarLanderContinuous-v2')

        winner_net = self.create_net(self.winner)

        ob = env.reset()
        while True:
            action = winner_net.activate(ob)
            ob, reward, done, info = env.step(np.array(action))
            env.render()
            if done:
                ob = env.reset()


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pole.ini'.format(NAME))

    w = PoleNeat('pole', config_path)

    w.train(300)
    w.play()


if __name__ == '__main__':
    main()
