# -*- coding: utf-8 -*-

#GENETIC PROGRAMMING FOR SYMBOLIC REGRESSION
#MODULE: individual.py
#AUTHOR: Pedro Henrique Ramos Costa
#
#Developed as part of a coursework for the module Natural Computing (Computacao Natural)
#at the Federal University of Minas Gerais (UFMG).
#This module describes the INDIVIDUALS of the population. Each individual
#is an expression tree and has its fitness (that means, how well that
#individual is fit inside that population). The fitness function is defined
#based on the premises of the coursework. 
#

#GLOBAL
from expression_tree import *
import traceback
from random import *
from copy import *

DEFAULT_MIN_DEPTH = 1
DEFAULT_MAX_DEPTH = 2
DEFAULT_NUM_VARS = 3

#INDIVIDUAL CLASS

class Individual:

	def __init__(self, max_depth=DEFAULT_MAX_DEPTH, min_depth=DEFAULT_MIN_DEPTH, size_vars=DEFAULT_NUM_VARS, full=False):
		self.tree = grow_tree(max_depth, min_depth, size_vars, full=full)
		self.max_depth = max_depth
		self.min_depth = min_depth
		self.size_vars = size_vars
		self.fitness = None

	def set_fitness(self, values_list, func_name):
		try:
			self.fitness = func_name(self, values_list)
		except:
		    print(traceback.format_exc())

	def mutation(self, mutation_prob):
		prob = random()
		if (prob < mutation_prob):
			#Sorts the depth of the node to be selected for mutation
			depth = rand_int(1, self.max_depth)
			affected_node = self.tree.get_subtree(depth)
			affected_node.mutate(self.size_vars, self.max_depth)

	def crossover(self, other_parent, crossover_prob):
		prob = random()
		if (prob <= crossover_prob):
			#Sorts the depth of the nodes to be selected for crossover
			depth = rand_int(1, self.max_depth)

			#make a copy of the parents, so they won't change the objects
			parent_1 = deepcopy(self)
			parent_2 = deepcopy(other_parent)

			#Select random subtree based on depth
			node_a = parent_1.tree.get_subtree(depth)
			node_b = parent_2.tree.get_subtree(depth)

			#Swaps the two nodes, which are at the same height			
			Node.swap_node(node_a, node_b)
			
			#The swapped parents are now the resulted children
			return parent_1, parent_2
		else:
			return None, None

	def __str__(self):
		ind = "\nNumber of variables: " + str(self.size_vars) + "\nTree:\n" + self.tree.__str__() + "\nFitness: " + str(self.fitness)

		return ind

#Testing routine
def crossover_test():
	population = []
	parent_1 = Individual()
	parent_2 = Individual()

	child_1, child_2  = parent_1.crossover(parent_2, 1.0)

	if child_1 != None:
		print "\nPai 1"
		print parent_1
		print "\nPai 2"
		print parent_2
		print "\nChild 1"
		print child_1
		print "\nChild 2"
		print child_2
	else:
		print "Crossover did not happen"

#Testing routine
def mutation_test():
	ind = Individual()
	print ind
	ind.mutation(1.0)
	print "After Mutation: "
	print ind

if __name__ == '__main__':
	crossover_test()
#	mutation_test()