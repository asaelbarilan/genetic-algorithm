''' genetic algorithm version :1 ; date: 24/01 10:07'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
xlsx = pd.ExcelFile('European Measurements.xlsx')
sheet1 = xlsx.parse(3) # Note the other datasets in the file.
variables = ["Month","Avg. Likes"]
data = sheet1.loc[:, variables].values


# Make the bias_selection function give more chances
# to solutions with better fitness values,
# but not necesseraly take the best ones, as currently done

def fitness_function(data, pop):
    fitness = []
    for solution in range(len(pop)):
        for event in range(len(data)):
            error = 0;
            event_time = data[event][0]
            event_measured = data[event][1]
            event_expected = pop[solution][0]*event_time**3 + pop[solution][1]*event_time**2 + pop[solution][2]*event_time + pop[solution][3]
            error +=(event_expected - event_measured)**2
        fitness.append(1/error) # We use 1/error in order to use a maximization mechanism, while we want to minimize the error
    return fitness

def biased_selection(population , fitness, num_parents):
    #original function
    sorted_fitness_args = np.argsort(fitness)
    n = sorted_fitness_args.shape[0]
    #new
    # df=pd.DataFrame(sorted_fitness_args)
    # best_samples=df.loc[-num_parents:].sample(frac=0.6, replace=True, random_state=1)
    # other_samples=df.loc[-num_parents:].sample(frac=0.6, replace=True, random_state=1)
    # population[sorted_fitness_args[-num_parents:],:]

    pop = []
    '''im taking a toss from -going over the range of fitness valus while streching the space to  the sum of fitness so
    the space is proportional to  all the fitness values. 
    than im taking random number in this range so i have higher chance to get larger numbers
    i creat treshold series which each treshold is the cumsum of fitness values so it is proportional to the range
    and from the difference between  the toss  and the treshold series i find the the right index in fitness and 
    from there the inde in the population form which i got the fitness value
    '''
    track=[]
    indexes=[]


    sorted_fitness_args = np.argsort(fitness)
    while len(pop)<250:
        toss = np.random.uniform(min(fitness),sum(fitness))
        tresholds=np.cumsum(np.sort(fitness))
        diff=(np.ones_like(tresholds)*toss)-tresholds
        j=np.where(abs(diff)==min(abs(diff)))[0][0]+1
        if j>=len(fitness):
            j-=1
        if j in indexes:
            continue
        indexes.append(j)
        track.append(np.sort(fitness)[j])
        pop.append(np.array(population)[j])


    #f you want to see the distrbution of chosing 
    num_bins = 30
    plt.hist(track, num_bins, facecolor='blue', alpha=0.5)
    plt.hist(np.sort(fitness)[250:], num_bins, facecolor='green', alpha=0.5)
    plt.hist(np.sort(fitness), num_bins, facecolor='red', alpha=0.5)
    plt.show()
    return np.array(pop)
    # parents = np.empty((num_parents, pop.shape[1]))
    # for parent_num in range(num_parents):
    #     max_fitness_idx = np.where(fitness == np.max(fitness))
    #     max_fitness_idx = max_fitness_idx[0][0]
    #     parents[parent_num, :] = pop[max_fitness_idx, :]
    #     fitness[max_fitness_idx] = -99999999999
    # return parents

# Make the recombination
# function make real random recombination along the chromosome and between chromosomes
def recombination(parents, offspring_size):
    offspring = np.empty(offspring_size)
    recombination_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:recombination_point] = parents[parent1_idx, 0:recombination_point]
        offspring[k, recombination_point:] = parents[parent2_idx, recombination_point:]
    return offspring

def mutation(offspring_recombination):
    for idx in range(offspring_recombination.shape[0]):
        random_value = np.random.randint(-100, 100, 1)
        random_index = np.random.randint(0,offspring_recombination.shape[1],1)
        offspring_recombination[idx, random_index] = offspring_recombination[idx, random_index] + random_value
    return offspring_recombination

# GA Parameters
formula_degree = 4
number_of_solutions = 500
number_of_parents = 250
population_size = (number_of_solutions,formula_degree)
number_of_generations = 20
best_outputs = []

# Genesis
new_population = np.random.randint(low=0, high=10000, size=population_size)
print("The population of the first generation: ")
print(new_population)

# Evolution
print ("\nEvolution:")
for generation in range(number_of_generations):

    fitness = fitness_function(data, new_population)
    print("Generation = ", generation, "\tBest fitness = ", round(1/np.max(fitness),5))
    # best_outputs.append(np.max(np.sum(new_population*formula_degree, axis=1)))
    best_outputs.append(round(1/np.max(fitness),5))
    parents = biased_selection(new_population, fitness, number_of_parents)
    offspring_recombination = recombination(parents, offspring_size=(population_size[0]-parents.shape[0], formula_degree))
    offspring_mutation = mutation(offspring_recombination)
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Results
print("\nThe population of the last generation: ")
print(new_population)
fitness = fitness_function(data, new_population)
best_match_idx = np.where(fitness == np.max(fitness))
print("Best solution: ", new_population[best_match_idx, :])

# Chart
plt.plot(best_outputs)
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.show()
