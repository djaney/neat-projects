import neat
import pickle
import gzip


class SaveWinner(neat.reporting.BaseReporter):
    def __init__(self, filename):
        self.filename = filename

    def post_evaluate(self, config, population, species, best_genome):
        with gzip.open(self.filename, 'w', compresslevel=5) as f:
            pickle.dump(best_genome, f, protocol=pickle.HIGHEST_PROTOCOL)


class Neat:
    def __init__(self, name, config_file, checkpoint=None):
        self.name = name
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)

        self.best_filename = 'checkpoints/' + self.name + '-best'
        self.checkpoint_filename = 'checkpoints/' + self.name + '-'

        self.winner = None
        if checkpoint is None:
            self.population = neat.Population(self.config)
        else:
            self.population = neat.Checkpointer.restore_checkpoint('checkpoints/' + checkpoint)

        # Add a stdout reporter to show progress in the terminal.
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix=self.checkpoint_filename))
        self.population.add_reporter(SaveWinner(self.best_filename))

    def train(self, generations):
        self.winner = self.population.run(self.eval_genomes, generations)
        return self.winner

    def create_net(self, genome):
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

    def eval_genomes(self, genomes, config):
        raise NotImplemented()

    def play(self):
        with gzip.open(self.best_filename) as f:
            winner = pickle.load(f)

        self.play_winner(winner)

    def play_winner(self, winner):
        raise NotImplemented()
