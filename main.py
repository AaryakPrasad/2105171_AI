import numpy
import imageio.v2 as imageio
import converter
import pygad
import matplotlib.pyplot

# Reading target image to be reproduced using Genetic Algorithm (GA).
target_im = imageio.imread('Pepsi.jpg')
target_im = numpy.asarray(target_im/255, dtype=numpy.float64)

# Target image after enconding. Value encoding is used.
target_chromosome = converter.img2chromosome(target_im)
print(target_chromosome)


def fitness_fun(ga_instance, solution, solution_idx):

    fitness = numpy.sum(numpy.abs(target_chromosome-solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness


def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


ga_instance = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0,
                       init_range_high=1,
                       mutation_probability=0.01,
                       parent_selection_type="sss",
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=on_gen)

ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

result = converter.chromosome2img(solution, target_im.shape)
matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("Using PyGAD for Reproducing Images")
matplotlib.pyplot.show()
