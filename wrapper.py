import neat


class Neat:
    def __init__(self, name, config_file, checkpoint=None):
        self.name = name
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        self.winner = None
        if checkpoint is None:
            self.population = neat.Population(self.config)
        else:
            self.population = neat.Checkpointer.restore_checkpoint('checkpoints/' + checkpoint)

        # Add a stdout reporter to show progress in the terminal.
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/' + self.name + '-'))

    def train(self, generations):
        self.winner = self.population.run(self.eval_genomes, generations)
        return self.winner

    def create_net(self, genome):
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

    def eval_genomes(self, genomes, config):
        raise NotImplemented()

    def play(self):
        raise NotImplemented()
