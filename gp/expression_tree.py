# -*- coding: utf-8 -*-

#GENETIC PROGRAMMING FOR SYMBOLIC REGRESSION
#MODULE: expression_tree.py
#AUTHOR: Pedro Henrique Ramos Costa
#
#Developed as part of a coursework for the module Natural Computing (Computacao Natural)
#at the Federal University of Minas Gerais (UFMG).
#This module describes the structures used to parse an expression tree, which is going
#to be used to describe the INDIVIDUALS in the generations during the genetic
#program. The individual class is in the module individual.py, but it basically contains
#an expression tree defined within this module and some other attributes.
#

#GLOBAL

from random import *
from math import *
from copy import *
from decimal import *

OPERATOR_TYPE = "Operator"
CONSTANT_TYPE = "Constant"
VARIABLE_TYPE = "Variable"
FUNCS = {'ADD' : 1, 'SUB' : 2, 'MUL' : 3, 'DIV' : 4, 'SEN' : 5, 'COS' : 6}
VARS = {'X' : 1}
SIZE_VARS = len(VARS)
SIZE_FUNCS = 4
MAX_DEPTH = 6
MIN_DEPTH = 3

#NODE CLASS
#
#The Class NODE has got 3 possible types:
#
#OPERATOR: 	+ (Add)
#			- (Sub)
#			* (Mul)
#			/ (Div)
#			sin (Sin)
#			cos (Cos)
#
#	Because of the fact that trigonometric functions take only
#	one parameter, by convention, whenever the node is a trigonometric
#	function, the LEFT NODE value is the parameter, and the 
#	RIGHT NODE value is set to 0
#
#CONSTANT: Any terminal which is not a variable nor an operator,
# i. e. , a float number
#
#VARIABLE:	Any literal which represents a variable. Code-wise, it is simply
#			an integer number between 1 ... N, as N represents the number of
#			variables in the function described by the tree.

class Node:

	def __init__(self, node_type, value, height, node_left=None, node_right=None):
		self.type = node_type
		self.value = value
		self.height = height
		self.node_left = node_left
		#Case when SEN and COS are in play 
		if ((node_type != OPERATOR_TYPE) or (value < 5)):
			self.node_right = node_right
		else:
			self.node_right = Node(CONSTANT_TYPE, 0.0, height+1)

	def __eval__(self, var_list_values):
		if (self.type == CONSTANT_TYPE):
			return self.value
		elif (self.type == OPERATOR_TYPE):
			return self.get_func(self.value, self.node_left.__eval__(var_list_values), self.node_right.__eval__(var_list_values))
		elif (self.type == VARIABLE_TYPE):
			return var_list_values[self.value - 1]

	def get_func(self, operator, value_left, value_right):
		if (operator == FUNCS['ADD']):
			return value_left + value_right
		if (operator == FUNCS['SUB']):
			return value_left - value_right
		if (operator == FUNCS['MUL']):
			return value_left * value_right
		if (operator == FUNCS['DIV']):
			if value_right == 0: 	#Division by zero is 1 by default
				return 1
			else:
				return value_left / value_right
		if (operator == FUNCS['SEN']):
			return sin(value_left)
		if (operator == FUNCS['COS']):
			return cos(value_left)

	def __str__(self, level = 0):
		ret = "|---"*level
		if self.type == OPERATOR_TYPE:
			ret += str(get_symbol(self.value, FUNCS))
		elif self.type == VARIABLE_TYPE:
			ret += "VAR" + str(self.value)
		else:
			ret += str(self.value)
		ret += "\n"
		if (self.node_left != None):
			ret += "L" + str(self.node_left.height) + ": " + self.node_left.__str__(level + 1)
		if (self.node_right != None):
			ret += "R" + str(self.node_right.height) + ": " + self.node_right.__str__(level + 1)

		return str(ret)

	#Returns a subtree from the desired depth. If the desired depth is not
	#possible (the tree is shorter than the desired depth), so it
	#returns the max possible subtree reached (a leaf). The parameter
	#full is always set to True so it will return for sure the maximum
	#possible subtree. If it's set to false, then it tries to find a node
	#with that depth iterately.
	def get_subtree(self, depth, full=True):
		if self.height == depth:
			return self
		else:
			#sort if we're going through the left path or the right path
			#0 - Left
			#1 - Right
			path = rand_int(0, 1)
			if (path == 0):
				if (self.node_left != None):
					return self.node_left.get_subtree(depth)
				else:
					if (self.node_right != None):
						return self.node_right.get_subtree(depth)
					else:
						return self
			elif (path == 1):
				if (self.node_right != None):
					return self.node_right.get_subtree(depth)
				else:
					if (self.node_left != None):
						return self.node_left.get_subtree(depth)
					else:
						return self

	def mutate(self, vars_size, max_depth):
		#When a node mutates, it has 3 possibilites:
		#0 - It's going to become a Constant:
		#	In this case, if it's not a leaf, their children
		#	are set to None.
		#1 - It's going to become a Variable:
		#	Same above applies. We need the number of variables
		#	to know how many values are possible here.
		#2 - It's going to become a Function:
		#	We also need the max depth of the tree to know which
		#	is the maximum depth of the new subtree generated to
		#	replace the affected node.
		mutation_type = rand_int(0, 2)
		if mutation_type == 0:
			self.type = CONSTANT_TYPE
			self.value = random()
			self.node_left = None
			self.node_right = None
		elif mutation_type == 1:
			self.type = VARIABLE_TYPE
			self.value = rand_int(1, vars_size)
			self.node_left = None
			self.node_right = None
		elif mutation_type == 2:
			#In this case, if the depth of the node is already the
			#max depth, this method below will return a leaf anyway.
			new_subtree = grow_tree(max_depth, depth=self.height)
			Node.swap_node(self, new_subtree)

	#THIS IS ONLY USED WHEN THE EXPRESSION TREE HAS GOT ONLY ONE VARIABLE, 
	#that means this method is only useful to generate a set of coordinates pairs
	#in order to generate/plot a chart of this tree
	def get_XY_coordinates_func(self, values_list):
		coords = []
		for values in values_list:
			Y = self.__eval__(values)
			X = values[0]
			coords.append([X, Y])

		return coords

	#Use it carefully. Always remember to deepcopy the object otherwise you'll lose information
	@staticmethod
	def swap_node(node_a, node_b):
		node_a.__dict__, node_b.__dict__ = node_b.__dict__, node_a.__dict__

def get_symbol(val, collection):
	for symbol, value in collection.items():
		if value == val:
			return symbol

#Returns a random Integer between two values
def rand_int(a, b):
	if b < a:
		return randint(b, a)
	return randint(a, b)

def grow_tree(max_depth=MAX_DEPTH, min_depth=MIN_DEPTH, size_vars=SIZE_VARS, depth=0, full=False):
	#If depth is MAX, then it is the end of the tree.
	#method leaf() returns either a CONSTANT or a VARIABLE.
	if (depth == max_depth):
		return leaf(size_vars, depth)
	#It has 50% of chance of either being a LEAF or another tree
	else:	
		type_num = rand_int(0, 1)
		#If type_num is 0, then it is going to be a LEAF
		#as long as the min_depth has  been surpassed and
		#the tree is not FULL
		if type_num == 0 and depth >= min_depth and full == False:	
			return leaf(size_vars, depth)
		#Then it is going to be another tree (an operator)
		#In this case, we have to decide which operator it'll be
		else:	
			operator_num = rand_int(1, SIZE_FUNCS)
			node_left = grow_tree(max_depth, min_depth, size_vars, depth + 1, full)
			node_right = grow_tree(max_depth, min_depth, size_vars, depth + 1, full)
			tree = Node(OPERATOR_TYPE, operator_num, depth, node_left, node_right)
			return tree

	return "Exception"

def leaf(size_vars, depth):
	#Is it going to be a constant or a variable?
	#	1 - Constant
	#	2 - Variable
	type_num = rand_int(0, 1)
	#it is a constant
	if type_num == 0:	
		leaf = Node(CONSTANT_TYPE, random(), depth)
		return leaf
	#it is a variable. In this case, we have to decide which variable
	else:	
		var_num = rand_int(1, size_vars)
		leaf = Node(VARIABLE_TYPE, var_num, depth)
		return leaf

#testing routine
def test_tree():
	swap_height = rand_int(1, MAX_DEPTH)

	tree_grow = grow_tree(size_vars=6)
	print tree_grow
	value = tree_grow.__eval__([1,1,1,1,1,1,])
	print "\nEVAL: "+ str(value)
	subtree_grow = tree_grow.get_subtree(swap_height)
	subtree_grow_copy = deepcopy(subtree_grow)

	tree_full = grow_tree(min_depth=MAX_DEPTH, size_vars=6, full=True)
	print tree_full
	value = tree_full.__eval__([1,1,1,1,1,1,])
	print "\nEVAL: "+ str(value)
	subtree_full = tree_full.get_subtree(swap_height)
	subtree_full_copy = deepcopy(subtree_full)

	Node.swap_node(subtree_full, subtree_grow)

	print "\nSWITCH HEIGHT: " + str(swap_height)

	print "\nSUBTREE GROW: "
	print subtree_grow_copy

	print "\nNOVA FULL:\n"
	print tree_full
	
	print "\nSUBTREE FULL: "
	print subtree_full_copy

	print "\nNOVA GROW:\n"
	print tree_grow

def test_mutation():
	tree_grow = grow_tree(size_vars=6)
	print "Tree now: \n"
	print tree_grow
	value = tree_grow.__eval__([1,1,1,1,1,1,])
	print "\nEVAL: "+ str(value) + "\n"

	node_m = tree_grow.get_subtree(2)

	print "\n Node selected: "
	print node_m

	node_m.mutate(6, MAX_DEPTH)

	print "\n Node mutated: "
	print node_m

	print "\n Tree after mutation: "
	print tree_grow
	value = tree_grow.__eval__([1,1,1,1,1,1,])
	print "\nEVAL: "+ str(value) + "\n"

if __name__ == '__main__':
	test_mutation()