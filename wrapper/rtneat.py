from wrapper.neat import Neat
from neat.six_util import iteritems, iterkeys, itervalues
import threading
import numpy as np


class IntervalThread(threading.Thread):
    def __init__(self, event, threaded_function):
        threading.Thread.__init__(self)
        self.stopped = event
        self.threaded_function = threaded_function

    def run(self):
        while not self.stopped.wait(0.1):
            self.threaded_function()


class RtNeat(Neat):

    rt_population_nn = {}

    def __init__(self, name, config_file):
        super().__init__(name, config_file)
        self.build_population_nn()

    def eval_genomes(self, genomes, config):
        pass

    def set_fitness(self, population_index):
        pass

    def build_population_nn(self):
        nn = {}
        for index, pop in list(iteritems(self.population.population)):
            nn[index] = self.create_net(pop)
        self.rt_population_nn = nn

    def __get_rt_iterate_function(self):
        return lambda: self.rt_iterate()

    def rt_iterate(self):
        self.population.reporters.start_generation(self.population.generation)

        # Gather and report statistics.
        best = None
        worst = None
        for g in itervalues(self.population.population):
            if not hasattr(g, 'birth'):
                g.birth = self.population.generation

            if g.fitness is None:
                continue

            if best is None or g.fitness > best.fitness:
                best = g
            if (worst is None or g.fitness < worst.fitness) and \
                    self.population.generation - g.birth > self.population.config.stagnation_config.max_stagnation:
                worst = g

        population_with_fitness = dict([p for p in iteritems(self.population.population) if p[1].fitness is not None])
        if population_with_fitness:
            self.population.reporters.post_evaluate(self.config, population_with_fitness, self.population.species, best)

        # Track the best genome ever seen.
        if self.population.best_genome is None or best.fitness > self.population.best_genome.fitness:
            self.population.best_genome = best

        if not self.population.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = self.population.fitness_criterion(g.fitness for g in itervalues(population_with_fitness))
            if fv >= self.population.config.fitness_threshold:
                self.population.reporters.found_solution(self.population.config, self.population.generation, best)
                return

        # remove worst genome and replace with new one
        if worst is not None:
            new_genomes = self.population.reproduction.reproduce(self.population.config, self.population.species, 1,
                                                                self.population.generation)
            new_key = np.max([g.key for g in itervalues(self.population.population)]) + 1
            for g in itervalues(new_genomes):
                del self.population.population[worst.key]
                g.key = new_key
                self.population.population[new_key] = g
                print(new_key)
                break

        # Divide the new population into species.
        if population_with_fitness:
            self.population.species.speciate(self.population.config, population_with_fitness,
                                             self.population.generation)

        # self.population.reporters.end_generation(self.population.config, self.population.population,
        #                                          self.population.species)

        self.population.generation += 1

        # if self.population.config.no_fitness_termination:
        #     self.population.reporters.found_solution(self.population.config, self.population.generation, self.population.best_genome)

    def rt_get_population_ids(self):
        return list(iterkeys(self.population.population))

    def rt_activate(self, genome_id, inputs):
        genome = self.rt_population_nn.get(genome_id)

        if genome is not None:
            return genome.activate(inputs)
        else:
            return None

    def rt_set_fitness(self, genome_id, fitness):
        genome = self.population.population.get(genome_id)
        if genome is not None:
            genome.fitness = fitness