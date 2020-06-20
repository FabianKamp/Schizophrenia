from deap import creator, base, tools
import random
import IndividualFit

# create individuals
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox() # Deap syntax: Container where all the elements and operations are "registered"

# set initial parameters of WC model
# The parameters for the first generation of individuals. Set range in which parameters should be set.
allParams = {'K_gl':[2,4], 'signalV':[15,25], 'sigma_ou':[0,1]}

def initRandParam(individual, allParams):
    RandParams = []
    for key, value in allParams.items():
        rparameter = random.uniform(value[0], value[1])
        RandParams.append(rparameter)
    Individual = individual(RandParams)
    return Individual

toolbox.register("individual", initRandParam, creator.Individual, allParams)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # n is defined in main function

# Genetic Operations
# Register genetic operators with default arguments in toolbox
toolbox.register("evaluate", IndividualFit.getFit) # Evaluate Function
toolbox.register("mate", tools.cxTwoPoint) # Mating function
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Evolving the Population
# Algorithm that combine all the individual parts and performs the evolution
# until the One Max problem is solved.
# This is done in a function called main()

# Creating the Population
def main():
    pop = toolbox.population(n=3)     # n was not defined during the initialization but now
    # pop is a list of 300 individuals.

    # Evaluate the fitness of the population
    fitnesses = list(map(toolbox.evaluate, pop))    # evaluate fitness for each individual
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)                    # assign fitness value to each individual in the pop

    CXP = 0.5 # Probability for crossing of individual
    MUTP = 0.2 # Probability for mutation

    # Performing the Evolution
    fits = [ind.fitness.values[0] for ind in pop] # list all fitness values

    g = 0 # Counter for the number of generations
    while g < 3:
        # Each iteration is a new generation
        g += 1
        print(f"Generation {g}")

        offspring = toolbox.select(pop, len(pop)) # select all ind from population
        offspring = list(map(toolbox.clone, offspring)) # clones all offsprings
        # It is important to clone the offspring to get a copy of the population
        # to not change the original population but only the copy

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXP:
                toolbox.mate(child1, child2)
                del child1.fitness.values # Crossover mating yields two children
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTP:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Replace the old population by the offspring
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    return pop

pop = main()
print(max(pop))
