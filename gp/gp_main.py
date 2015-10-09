# -*- coding: utf-8 -*-

#GENETIC PROGRAMMING FOR SYMBOLIC REGRESSION
#MODULE: gp_main.py
#AUTHOR: Pedro Henrique Ramos Costa
#
#Developed as part of a coursework for the module Natural Computing (Computacao Natural)
#at the Federal University of Minas Gerais (UFMG).
#This is the main module of the genetic program, which is going to use
#genetci programming concepts to symbolic regression. In this module
#the user must define the FITNESS method (that means, the way each
#individual is going to be evaluated every generation), as well as
#the input file with the data (values for both variables and result
#obtained experimentally).
#
#If one wants to change the fitness function, it must implement
#it here and pass it as a parameter to the fitness() method inside
#the Individual class (in the individual.py module).
#
#Data concerning the number of variables is obtained from the file
#directly. The file must have a specific format with each line containing
#a different set of variable values and expected result for
#the function which is trying to be discovered. It can be a simple
#text file like this:
#
# PARAM_1 PARAM_2 ... PARAM_N RESULT
# PARAM_1 PARAM_2 ... PARAM_N RESULT
# ...
#
#And so on. In which PARAM_1, ..., PARAM_N are all the variables
#values, like x, w, t... And the LAST parameter on each line is the f(x, w, t...)
#result for those previous values. That means the last column of the file
#must contain the expected result for the correct function to be
#discovered, given that set of values previous in line.
#

#GLOBAL

import expression_tree
import individual
from random import *
from math import *
from decimal import *
from copy import *
from numpy import average
import datetime
import matplotlib.pyplot as plt
import time
import numpy as np 

#=================================
#GENETIC PROGRAMMING PARAMETERS:
#=================================
#Play with them as you run it
#until you find the best setting
#for your instance.
#=================================

#If the user wants to make no change at all it can run this algorithm
#by simply setting the input file to the desired one, as long as
#the file is formatted as specified above.
FNAME_ = "SR_div.txt"
INPUT_FILE_NAME = "../datasets/" + FNAME_

#Directory to output the results
OUTPUT_DIR = "../exec_logs/sr_div/"

#Number of executions. How many times do you want to execute the program with
#this seed on this same file?
NUM_EXECS = 10

#Obs: Depending on the complexity of the function you're trying to 
#approach by regression, the depth of the tree must not be a big number
#For simple functions like Y = x^2 , or z = x + y, the best option
#would be starting from a short tree.
INDIVIDUAL_MAX_DEPTH = 5

#The Min_depth is utilized in the method that generate trees
#with random depth. It is important that these trees depths are
#contained in a limited space. The population is generated
#using ramped half-half, so 50% of it are full trees, and 50%
#are irregular trees.
INDIVIDUAL_MIN_DEPTH = 1

#Pretty much self-explanatory. Probability of crossover happening
#between the selected couples each generation.
CROSSOVER_PROB = 0.9

#Same as above, but concerning mutation. Probability of mutation
#happening on each individual every generation.
MUTATION_PROB = 0.05

#Defines the size of the tournament. Increase it to increase
#the seletive pressure.
TOURNAMENT_SIZE = 2

#Size of the population...
POP_SIZE = 500

#Number of individuals that are going to PERSIST
#in the poulation after the generation. This parameter
#defines how elitist your program will be. Attention:
#if you increase the number of individuals that will persist
#every gen, the odds that you're algorithm converges are
#a lot more likely to increase as well. Change it wisely
NUM_ELITE_INDIVIDUALS = 1

#Numer of generations intended to run the program
NUM_GENS = 100

#Miminal optimal fitness. If the fitness is below it, the
#program halts.
MINIMAL_OPTIMAL = 0.0

#The following variables will be used to log the results of the algorithm
BEST_INDV = None
BEST_FITNESS = 0.0
WORST_FITNESS = 0.0
AVG_FITNESS = 0.0
BEST_EVO = []
AVG_EVO = []
WORST_EVO = []
BETTER_CHILDREN_EVO = []
LAST_GEN = 0.0
NUMBER_CROSSOVERS = 0.0
BETTER_CHILDREN = 0.0
BETTER_CHILDREN_PROPORTION = 0.0
NUM_VARS = 1
LOG = []
ELAPSED_TIME = 0.0

#FITNESS:
#
#The fitness function is given considering the absolute error
#between the vaue obtained from the individual expression and
#the valued expected. It is given a list of many instances with
#all variables and expected results for each instance. Each instance
#is a list with variables values and the expected output of
#the correct function, like this:
#
#  [x1, x2, x3, ..., xN, f(x)]
#
#In which f(x) is the result expected in the correct function which
#this GP tries to approach
#
def FITNESS_BY_ABS_ERROR(ind, values_list):
	abs_error = 0.0
	#Each item of values_list is another list with X1, X2, X3, ..., f(X)
	num_vars = len(values_list[0])

	for values in values_list:
		#Evaluates it with only the values of the variables, so taking
		#away the last value of the list, which is f(x)
		evaluated_result = ind.tree.__eval__(values[:num_vars-1])
		expected_result = values[num_vars-1]
		abs_error += abs(evaluated_result - expected_result)

	return abs_error

def get_values_list(filename=INPUT_FILE_NAME):
	file_ = open(filename, 'rb')
	values_list = []
	for row in file_:
		values_ = []
		values = row.split()
		for value in values:
			val_float = float(value)
			values_.append(val_float)
		values_list.append(values_)

	return values_list

def gen_population(size=POP_SIZE):
	half_size = int(size / 2)
	population = []

	#Gen first half: GROW method
	for i in range(0, half_size):
		new_ind = individual.Individual(max_depth=INDIVIDUAL_MAX_DEPTH, min_depth=INDIVIDUAL_MIN_DEPTH, size_vars=NUM_VARS)
		population.append(new_ind)

	if size%2 != 0:
		half_size += 1

	#Gen second half: FULL method
	for i in range(0, half_size):
		new_ind = individual.Individual(max_depth=INDIVIDUAL_MAX_DEPTH, size_vars=NUM_VARS, full=True)
		population.append(new_ind)

	return population

#Change here the fitness function parameter as you
#desire another fitness evaluation method
def evaluate_fitness(population, values_list, func_name=FITNESS_BY_ABS_ERROR):
	for ind in population:
		ind.set_fitness(values_list, func_name)

def set_fitness_statistics(population):
	pop_sorted = sorted(population, key=lambda y: y.fitness)
	global BEST_FITNESS, BEST_INDV, WORST_FITNESS, AVG_FITNESS, BETTER_CHILDREN_PROPORTION, \
		BETTER_CHILDREN, NUMBER_CROSSOVERS, BEST_EVO, WORST_EVO, AVG_EVO, BETTER_CHILDREN_EVO
	
	#set each generation statistics
	BEST_FITNESS = pop_sorted[0].fitness
	BEST_INDV = pop_sorted[0]
	WORST_FITNESS = pop_sorted[-1].fitness
	AVG_FITNESS = average([ind.fitness for ind in pop_sorted])
	if NUMBER_CROSSOVERS > 0.0:
		BETTER_CHILDREN_PROPORTION = (BETTER_CHILDREN / (NUMBER_CROSSOVERS*2.0)) * 100

	#set evolution statistics
	BEST_EVO.append(BEST_FITNESS)
	AVG_EVO.append(AVG_FITNESS)
	WORST_EVO.append(WORST_FITNESS)
	BETTER_CHILDREN_EVO.append(BETTER_CHILDREN_PROPORTION)

def tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
	indexes=[]
	i = 0
	#Randomly select indexes from the population
	while i < tournament_size:
		index = expression_tree.rand_int(0, POP_SIZE-1)
		if index not in indexes:
			i += 1
			indexes.append(index)

	#Append the random selected individuals to the tournament
	tournament = []
	for index in indexes:
		tournament.append(population[index])

	tournament.sort(key=lambda x: x.fitness)

	#Returns the best individuals from the tournament
	return tournament[0]

def run(num_gens=NUM_GENS):
	global NUMBER_CROSSOVERS, BETTER_CHILDREN, LAST_GEN, NUM_VARS, ELAPSED_TIME
	BETTER_CHILDREN = 0.0
	LAST_GEN = 0
	print "EXECUTANDO ALGORITMO DE PROGRAMACAO GENETICA, AGUARDE..."

	#Get the set of values from file
	values_list = get_values_list()
	NUM_VARS = len(values_list[0]) - 1

	#Set the log file
	write_to_log("File: " + str(INPUT_FILE_NAME))
	write_to_log("N_vars: " + str(NUM_VARS) + " N_gens: " + str(NUM_GENS) + \
		" Pop_size: " + str(POP_SIZE) + " XP: " + str(CROSSOVER_PROB) + " MP: " + \
		str(MUTATION_PROB) + " TS: " + str(TOURNAMENT_SIZE) + " MAX_D: " + str(INDIVIDUAL_MAX_DEPTH) + \
		" MIN_D: " + str(INDIVIDUAL_MIN_DEPTH))
	write_to_log("[GEN], [BEST_FIT], [WORST_FIT], [AVG_FIT], [% BETTER CHILDREN]")

	#Setup timer
	start = time.time()

	#Generate first generation of individuals
	#and evaluate their fitness, as it gets the
	#statistics. 
	population = gen_population()
	evaluate_fitness(population, values_list)
	set_fitness_statistics(population)
	# print_gen()
	log_()

	if BEST_FITNESS < MINIMAL_OPTIMAL:
		print_gen()
		print "\nBEST INDIVIDUAL FOUND: "
		print BEST_INDV
		return

	#Now, for each generation after the first one:
	i = 1
	while i < num_gens:
		#New population for next generation
		new_population = []
		best = deepcopy(BEST_INDV)

		#Try to make crossovers between the individuals
		NUMBER_CROSSOVERS = 0.0
		BETTER_CHILDREN = 0.0
		for j in range(0, POP_SIZE/2):
			#SELECTION: select the parents in a tournament
			parent_1 = tournament_selection(population)
			parent_2 = tournament_selection(population)

			#CROSSOVER: try to make the crossover between two individuals
			child_1, child_2 = parent_1.crossover(parent_2, CROSSOVER_PROB)
			#if the crossover happens...
			if ((child_1 != None) and (child_2 != None)):
				NUMBER_CROSSOVERS += 1
				evaluate_fitness([child_1, child_2], values_list)
				family = [child_1, child_2, parent_1, parent_2]
				family.sort(key=lambda y: y.fitness)
				#get information about the amount of children that were better than their parents
				if family[0] == child_1 or family[0] == child_2:
					BETTER_CHILDREN += 1.0
				if family[1] == child_1 or family[1] == child_2:
					BETTER_CHILDREN += 1.0
				#only the best 2 individuals from the family enter the new population
				if family[0] not in new_population:
					new_population.append(family[0])
				if family[1] not in new_population:
					new_population.append(family[1])

		#MUTATION: now, before we evaluate each ind from the new population, there's a chance
		#mutation will happen.
		for ind in new_population:
			ind.mutation(MUTATION_PROB)

		#print len(new_population)

		#best individual from last gen go to new pop
		new_population.append(best)

		#after all crossovers that could possibly happen happened, it is time to complete
		#the new population with the remaining slots. It will be used the grow method again
		complete_pop_ = gen_population(POP_SIZE - len(new_population))
		new_population.extend(complete_pop_)

		#EVALUATION:
		evaluate_fitness(new_population, values_list)
		set_fitness_statistics(new_population)
		LAST_GEN = i
		log_()
		#print_gen()

		#HALTING CRITERIA
		if BEST_FITNESS <= MINIMAL_OPTIMAL:
			end = time.time()
			ELAPSED_TIME = end - start
			break

		population = new_population
		i += 1

	#finish timer
	end = time.time()
	ELAPSED_TIME = end - start

	#At the end, print the best individual found
	print "\nBEST INDIVIDUAL FOUND ON: "
	print_gen()
	print BEST_INDV
	
	filename = OUTPUT_DIR + FNAME_ + "_{:(%m-%d-%Y)_at_%Hh-%Mm-%Ss}.txt".format(datetime.datetime.now())
	log_execution(filename)
	
	#=>>>>>>THERE'S A BUG HERE IF YOU TURN THIS ON!!!!
	#ONLY TURN THIS ON IF THE FUNCTION YOU'RE TRYING TO FIND HAS ONLY 1 VARIABLE, OTHERWISE
	#MATPLOTLIB WON'T BE ABLE TO PLOT IT FOR YOU IN A 2D SPACE!!!!!!!
	gen_best_ind_chart(values_list, filename)
	
	#Generate a chart with the evolution of your program.
	gen_evolution_chart(filename, BEST_EVO, AVG_EVO, WORST_EVO, BETTER_CHILDREN_EVO)

def write_to_log(str):
	global LOG
	LOG.append(str)

#Log of the generations to be written to a file at the end of the execution.
#The numbers were all rounded to 3 decimal points.
def log_():
	str_ = str(LAST_GEN+1) + ", {0:.5f}".format(BEST_FITNESS) + ", {0:.5f}".format(WORST_FITNESS) + \
		", {0:.5f}".format(AVG_FITNESS) + ", {0:.3f}%".format(BETTER_CHILDREN_PROPORTION)
	write_to_log(str_)

def log_execution(filename):
	global ELAPSED_TIME, BEST_INDV, LOG
	file_out = open(filename, 'w')
	for row in LOG:
		file_out.write(str(row)+"\n")
	file_out.write(BEST_INDV.__str__())
	file_out.write("\nElapsed time: " + "{0:.6f} (s)".format(ELAPSED_TIME))
	print "Logged data and graphs on ../exec_logs/"
	LOG = []

def gen_best_ind_chart(values_l, filename):
	global BEST_INDV
	coords = BEST_INDV.tree.get_XY_coordinates_func(values_l)
	fig_ind, ax_ind = plt.subplots()
	
	#Plots the original dataset
	x2, y2 = zip(*values_l)
	ax_ind.plot(x2, y2, 'bo')

	#Plots the best individual
	x, y = zip(*coords)	
	ax_ind.plot(x, y, 'ro')

	ax_ind.axhline(y=0, color='k')
	ax_ind.axvline(x=0, color='k')
	ax_ind.grid(True, which='both')

	fig_ind.savefig(filename + ".plotted.png")


#This method plots a graph showing the evolution of the genetic program,
#considering the evolution of the fitness of the individuals in each
#generation.
def gen_evolution_chart(filename, best_evo, avg_evo, worst_evo, betchil_evo):
	fig_evo, ax_evo = plt.subplots(4)
	fig_evo.subplots_adjust(hspace=0.2)

	y1 = np.array(worst_evo)
	y2 = np.array(avg_evo)
	y3 = np.array(best_evo)
	y4 = np.array(betchil_evo)
	x1 = np.array(range(0, len(worst_evo)))
	x2 = np.array(range(0, len(avg_evo)))
	x3 = np.array(range(0, len(best_evo)))
	x4 = np.array(range(0, len(betchil_evo)))

	plt.setp(ax_evo[0].get_xticklabels(), visible=False)
	plt.setp(ax_evo[1].get_xticklabels(), visible=False)
	plt.setp(ax_evo[2].get_xticklabels(), visible=False)
	plt.setp(ax_evo[1].get_yticklabels(), visible=False)
	plt.setp(ax_evo[2].get_yticklabels(), visible=False)

	ax_evo[0].grid(True, which='both')
	ax_evo[1].grid(True, which='both')
	ax_evo[2].grid(True, which='both')
	ax_evo[3].grid(True, which='both')

	#plots the children proportion
	ax_evo[0].scatter(x4, y4, color = 'b', s=5.0)
	#plots the worst evo
	ax_evo[1].scatter(x1, y1, color = 'r', s=5.0)
	#plots the avg evo
	ax_evo[2].scatter(x2, y2, color = 'y', s=5.0)
	#plots the best evo
	ax_evo[3].scatter(x3, y3, color = 'g', s=5.0)

	fig_evo.savefig(filename + ".evolution.png")

def clear_statistic_buffers():
	global BEST_EVO, AVG_EVO, WORST_EVO, BETTER_CHILDREN_EVO
	BEST_EVO = []
	AVG_EVO = []
	WORST_EVO = []
	BETTER_CHILDREN_EVO = []

def gen_final_results(list_best, list_avg, list_worst, list_children):
	#The final results start as 4 empty lists
	final_best = []
	final_avg = []
	final_worst = []
	final_children = []

	best = [[row[i] for row in list_best] for i in range(0, NUM_GENS)]
	avg = [[row[i] for row in list_avg] for i in range(0, NUM_GENS)]
	worst = [[row[i] for row in list_worst] for i in range(0, NUM_GENS)]
	children = [[row[i] for row in list_children] for i in range(0, NUM_GENS)]
	
	for j in range(0, NUM_GENS):
		final_best.append(sum(best[j])/NUM_EXECS)
		final_avg.append(sum(avg[j])/NUM_EXECS)
		final_worst.append(sum(worst[j])/NUM_EXECS)
		final_children.append(sum(children[j])/NUM_EXECS)
	
	filename = OUTPUT_DIR + FNAME_ + "_final.txt"

	file_ = open(filename, 'w')
	file_str = "File: " + str(INPUT_FILE_NAME)
	str_ = "N_vars: " + str(NUM_VARS) + " N_gens: " + str(NUM_GENS) + \
		" Pop_size: " + str(POP_SIZE) + " XP: " + str(CROSSOVER_PROB) + " MP: " + \
		str(MUTATION_PROB) + " TS: " + str(TOURNAMENT_SIZE) + " MAX_D: " + str(INDIVIDUAL_MAX_DEPTH) + \
		" MIN_D: " + str(INDIVIDUAL_MIN_DEPTH)
	file_.write("\n" + file_str + "Final results.\n" + "SEED: " + str_)
	file_.write("\nARITHMETIC MEAN OF ALL EXECUTIONS\nGEN, BEST, WORST, AVG, %BETTER CHILDREN\n")
	for i in range(0, NUM_GENS):
		file_.write("\n" + str(i+1) + ", " + str(final_best[i]) + ", " + str(final_worst[i]) + ", " + str(final_avg[i]) + ", " + str(final_children[i]))

	gen_evolution_chart(filename, final_best, final_avg, final_worst, final_children)

#testing routines
def print_gen():
	print "\ngen " + str(LAST_GEN+1) + " BEST: " + str(BEST_FITNESS) + " WORST: " + \
		str(WORST_FITNESS) + " AVG: " + str(AVG_FITNESS) + " BETTER CHILDREN: " + str(BETTER_CHILDREN_PROPORTION) + "%\n" + \
		"Nº CROSSOVERS: " + str(NUMBER_CROSSOVERS) + " Nº CHILDREN: " + str(NUMBER_CROSSOVERS*2)

def test_fitness(values_list, func_name):
	pop = gen_population()
	for ind in pop:
		ind.set_fitness(values_list, func_name)

	parent_1, parent_2 = tournament_selection(pop)

	print "best parents: "

	print parent_1
	print parent_2

def print_log():
	global LOG
	for row in LOG:
		print row

def test_population():
	pop = gen_population(11)
	for p in pop:
		print p

if __name__ == '__main__':
	#This lists store the evolution of ALL the executions of the GP,
	#in order to make a final chart with the mean of every generation for
	#each execution.
	best_fitness_all = []
	avg_fitness_all = []
	worst_fitness_all = []
	b_children_propotion_all = []

	for i in range (0, NUM_EXECS):
		run()
		best_fitness_all.append(BEST_EVO)
		avg_fitness_all.append(AVG_EVO)
		worst_fitness_all.append(WORST_EVO)
		b_children_propotion_all.append(BETTER_CHILDREN_EVO)
		clear_statistic_buffers()

	#Now, after the N executions, we have 4 lists that
	#contains the data of the N executions, per generation.
	#We need to get now to get some statistics on them.
	gen_final_results(best_fitness_all, avg_fitness_all, worst_fitness_all, b_children_propotion_all)
