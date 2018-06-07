import neat
import os
import gym
import numpy as np

NAME = 'pole'


def eval_genomes(genomes, config):
    env = gym.make('LunarLanderContinuous-v2')

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        ob = env.reset()
        reward_sum = 0
        while True:
            action = net.activate(ob)
            ob, reward, done, info = env.step(np.array(action))
            reward_sum = reward_sum + reward
            if done:
                break
            genome.fitness = reward_sum


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    play(winner, config)


def play(winner, config):
    env = gym.make('LunarLanderContinuous-v2')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    ob = env.reset()
    reward_sum = 0
    while True:
        action = winner_net.activate(ob)
        ob, reward, done, info = env.step(np.array(action))
        reward_sum = reward_sum + reward
        if done:
            break


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '{}.ini'.format(NAME))
    run(config_path)
