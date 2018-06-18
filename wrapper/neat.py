import neat
import pickle
import gzip
import socket
import sys

class SaveWinner(neat.reporting.BaseReporter):
    def __init__(self, filename):
        self.filename = filename

    def post_evaluate(self, config, population, species, best_genome):
        with gzip.open(self.filename, 'w', compresslevel=5) as f:
            pickle.dump(best_genome, f, protocol=pickle.HIGHEST_PROTOCOL)


class Neat:
    def __init__(self, name, config_file, checkpoint=None, checkpoint_interval=20):
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
            self.population = neat.Checkpointer.restore_checkpoint(self.checkpoint_filename + checkpoint)

        # Add a stdout reporter to show progress in the terminal.
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(checkpoint_interval, filename_prefix=self.checkpoint_filename))
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
        print('Playing winner with fitness {}'.format(winner.fitness))
        self.play_winner(winner)

    def play_winner(self, winner):
        raise NotImplemented()

    def serve(self, host='localhost', port=9999, message_size=4096):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (host, port)
        print('starting up on {} port {}'.format(*server_address))
        sock.bind(server_address)

        while True:
            print('\nwaiting to receive message')
            data, address = sock.recvfrom(message_size)
            data = data.decode('utf-8')
            input_arr = data.split(' ')
            cmd = input_arr[0]
            params = input_arr[1:]

            # return population indexes
            if 'pop' == cmd:
                self.send(sock, ' '.join([str(s) for s in self.population.population.keys()]), address)
            if 'act' == cmd:
                winner_net = self.create_net(self.population.population.get(int(params[0])))
                inputs = [float(i) for i in params[1].split(',')]
                self.send(sock, ' '.join([str(s) for s in winner_net.activate(inputs)]), address)
            else:
                pass

    def send(self, sock, data, address):
        return sock.sendto(data.encode('utf-8'), address)
