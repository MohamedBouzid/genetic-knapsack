# Author : Mohamed BOUZID
# Knapsack Problem Solution using genetic algorithm


import random as rd
import numpy as np
import time
from bitarray import bitarray as ba

class Genetic:
  """ Abstract class for Genetic problems """
 
  def __init__(self, population_number, gene_number, sel_prob, mut_prob, iteration):
    """ Initialize Genetic Problem """
    self.population_number = population_number
    self.gene_number = gene_number
    self.sel_prob = sel_prob
    self.mut_prob = mut_prob
    self.iteration = iteration


  def initialize(self):
    """ Randomly initialize population 
        This Method must be reimplemented 
        according to the given problem """
    return np.array([])
	

  def mutate(self, chromosom, prob):
    """ Do a mutation in a chromosome """

    rand_prob = rd.random()
    clone = np.array(chromosom)
    if rand_prob<self.mut_prob:
      random_index = rd.randint(0, self.gene_number-1)
      clone[random_index] = not clone[random_index]
    return  clone
	

  def onepoint_crossover(self, parent1, parent2):
    """ Do a one point crossover between 2 parents """

    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      random_index = rd.randint(0, self.gene_number-1)
      parent1_tail = parent1[random_index:-1:1]
      #parent1_front = 
      parent2_tail = parent2[random_index:-1:1]
      np.delete( parent1,[i for i in range(random_index,len(parent1)-1)] )
      np.delete( parent2,[i for i in range(random_index,len(parent2)-1)] )
      parent1 = np.append(parent1,parent2_tail)
      parent2 = np.append(parent2,parent1_tail)
    return parent1,parent2


  def twopoint_crossover(self, parent1, parent2):
    """ Do a two point crossover between 2 parents """

    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      length = len(parent1)
      random_index1 = rd.randint(0, self.gene_number-1)
      random_index2 = rd.randint(0, self.gene_number-1)
      random1 = min(random_index1,random_index2)
      random2 = max(random_index1,random_index2)
      #parent 1
      parent1_tail = parent1[random2:length:1]
      parent1_front = parent1[0:random1:1]
      parent1_middle = parent1[random1:random2:1]
      #parent 2  
      parent2_tail = parent2[random2:length:1]
      parent2_front = parent2[0:random1:1]
      parent2_middle = parent2[random1:random2:1]
      child1 = np.append(np.append(parent2_front,parent1_middle),parent2_tail)
      child2 = np.append(np.append(parent1_front,parent2_middle),parent1_tail)
      return child1,child2
    else:
      return parent1,parent2

  def point_crossover(self, parent1, parent2):
    """ Do a every point crossover between 2 parents """
    #TODO
    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      length = len(parent1)
      child1 = np.array(parent1)
      child2 = np.array(parent2)
      for i in range(0,length,2):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
      return child1,child2
    else:
      return parent1,parent2

  def fitness(self, chromosom):
    """ Calculate the fitness of a chromosome 
        This Method must be reimplemented 
        according to the given problem """
    #TODO 
    return sum(chromosom)

  def pop_fitness(self, population):
    """ Calculate fitness for all population 
        This Method must be reimplemented 
        according to the given problem """
    #TODO
    return []

  def select_parent(self, population, fit):
    """ Select parent for crossover """
    sum_fitness = sum(fit)
    rand_sum = rd.randint(0,sum_fitness)
    s=0
    for i in range(15):
      if s<rand_sum:
        s+=fit[i]
      else:
        break   
    return population[i]

  def elite(self, population, fit):
    """ Get the elite of a population """
    max_index = fit.index(max(fit))
    fit.pop(max_index)
    return population[max_index],population[fit.index(max(fit))]

  def fittest(self, population, fit=None):
    """ Get the chromosom with the best fitness """
    if fit==None:
      fit = self.pop_fitness(population)
    max_fit = max(fit)
    return population[fit.index(max_fit)], max_fit


  def newGeneration(self, population, fit):
    """ Build new Generation 
        This Method must be reimplemented 
        according to the given problem """
    return np.array([])

class Knapsack(Genetic):
  """ Knapsack class problem implementation """

  def __init__(self, costs, weights, capacity, population_number, gene_number, mut_prob, sel_prob, iteration):
    """ Initialize Knapsack Problem """
    super().__init__(population_number, gene_number, sel_prob, mut_prob, iteration)
    self.costs = costs
    self.weights = weights
    self.capacity = capacity

  def initialize(self):
    """ Randomly initialize population """
    return np.array([ba(self.gene_number) for i in range(self.population_number)],float)
	

  def mutate(self, chromosom):
    """ Do a mutation in a chromosome """

    rand_prob = rd.random()
    clone = np.array(chromosom)
    if rand_prob<self.mut_prob:
      random_index = rd.randint(0, self.gene_number-1)
      clone[random_index] = not clone[random_index]
    return  clone
	

  def onepoint_crossover(self, parent1, parent2):
    """ Do a crossover between 2 parents """

    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      random_index = rd.randint(0, self.gene_number-1)
      parent1_tail = parent1[random_index:-1:1]
      #parent1_front = 
      parent2_tail = parent2[random_index:-1:1]
      np.delete( parent1,[i for i in range(random_index,len(parent1)-1)] )
      np.delete( parent2,[i for i in range(random_index,len(parent2)-1)] )
      parent1 = np.append(parent1,parent2_tail)
      parent2 = np.append(parent2,parent1_tail)
    return parent1,parent2


  def twopoint_crossover(self, parent1, parent2):
    """ Do a crossover between 2 parents """

    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      length = len(parent1)
      random_index1 = rd.randint(0, self.gene_number-1)
      random_index2 = rd.randint(0, self.gene_number-1)
      random1 = min(random_index1,random_index2)
      random2 = max(random_index1,random_index2)
      #parent 1
      parent1_tail = parent1[random2:length:1]
      parent1_front = parent1[0:random1:1]
      parent1_middle = parent1[random1:random2:1]
      #parent 2  
      parent2_tail = parent2[random2:length:1]
      parent2_front = parent2[0:random1:1]
      parent2_middle = parent2[random1:random2:1]
      child1 = np.append(np.append(parent2_front,parent1_middle),parent2_tail)
      child2 = np.append(np.append(parent1_front,parent2_middle),parent1_tail)
      return child1,child2
    else:
      return parent1,parent2

  def point_crossover(self, parent1, parent2):
    """ Do a crossover between 2 parents """

    rand_prob = rd.random()
    if rand_prob<self.sel_prob:
      length = len(parent1)
      child1 = np.array(parent1)
      child2 = np.array(parent2)
      for i in range(0,length,2):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
      return child1,child2
    else:
      return parent1,parent2

  def fitness(self, chromosom):
    """ Calculate the fitness of a chromosome """

    if capacity <sum(self.weights*chromosom):
      return 0
    return sum(self.costs*chromosom)

  def pop_fitness(self, population):
    """ Calculate fitness for all population """
    return [self.fitness(population[i]) for i in range(self.population_number)]

  def select_parent(self, population, fit):
    """ Select parent for crossover """
    sum_fitness = sum(fit)
    rand_sum = rd.randint(0,sum_fitness)
    s=0
    for i in range(15):
      if s<rand_sum:
        s+=fit[i]
      else:
        break   
    return population[i]

  def elite(self, population, fit):
    """ Get the elite of a population """
    max_index = fit.index(max(fit))
    fit.pop(max_index)
    return population[max_index],population[fit.index(max(fit))]

  def fittest(self, population, fit=None):
    """ Get the chromosom with the best fitness """
    if fit==None:
      fit = pop_fitness(population, self.costs, self.weights)
    max_fit = max(fit)
    return population[fit.index(max_fit)], max_fit


  def newGeneration(self, population, fit):
    """ Build new Generation """
    newGen = []
    pop_len = len(population)
    if pop_len % 2==1:
      newGen.append(elite(population, fit)[0])
    m1,m2 = self.elite(population, fit)
    newGen.append(m1)
    newGen.append(m2)
    for i in range(int(len(population)/2)-1):
      parent1 = self.select_parent(population, fit)
      parent2 = self.select_parent(population, fit)
      child1, child2 = self.twopoint_crossover(parent1, parent2)

      child1 = self.mutate(child1)
      child2 = self.mutate(child2)

      newGen.append(child1)
      newGen.append(child2)
    return np.array(newGen)

  def run(self):
    """ Running the problem """
    population = self.initialize()
    it = 0
    last_fittest = [1]*10
    fit = self.pop_fitness(population)
    f = 0
    while it<self.iteration:
      population = self.newGeneration(population, fit)
      fit = self.pop_fitness(population)
      last_fittest.pop(0)
      fi = max(fit) 
      if f<fi:
        f = fi 
      last_fittest.append(fi)
      print("popualtion : ",it+1," ## fittest : ",fi)
      it+=1

    f,m = self.fittest(population, fit)
    print("max fitness is = ",m)
    print("max fit selection = ",f)



start = time.time()

costs = np.array([ 825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261])
weights = np.array([382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684])
capacity = 6404180
population_number = 80
gene_number = len(costs)

knapsack = Knapsack(costs, weights, capacity, population_number, gene_number,0.9, 0.4,20000)

knapsack.run()

end = time.time()

print("execution time = ",end-start)




