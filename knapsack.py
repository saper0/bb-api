#######################################################################
# Author: Lukas Gosch
# Date: 21.1.2019
# Description:
# 	Implementation of an Instance class, using the b&b api, for the 0/1 
#	Knapsack problem.
#	
#	Usage: 1) Initialize a Knapsack object with a certain capacity,
#		   list of item values and weights (see constructor)
#		   2) Call its solve() method
#######################################################################

import numpy as np
from bb import Instance

class Knapsack(Instance):

	def __init__(self, capacity, values, weights, int_problem = False):
		""" Initialize a Knapsack problem and the initial solution. 
			
			capacity ... maximal capacity of the rucksack
			values ... array-like object storing the values of each item
			weights ... array-like object storing the weights of each item
			int_problem ... True if all values are integer numbers.
		"""
		# Problem specific data attributes
		self.max_capacity = np.array(capacity)
		self.values = np.array(values)
		self.weights = np.array(weights)
		self.vw_frac = self.values / self.weights
		# Get indices that would sort the value-weight fraction array (introsort)
		self.vw_frac_sortI_ascending = np.argsort(self.vw_frac, axis=-1, 
												  kind='quicksort')
		self.vw_frac_sortI = np.flip(self.vw_frac_sortI_ascending)
		self.n_items = self.values.shape[0]
		self.int_problem = int_problem
		
		# Instance specific data attributes
		self.solution = np.zeros((self.n_items,))
		self.index = 0
		self.curr_weight = 0
		self.curr_value = 0
		self.heur_solution = None
		self.heur_weight = None
		self.lower_bound = None
		self.upper_bound = None

	@classmethod
	def copy(cls, origin):
		""" Generate a new Knapsack instance based on a given instance.
			
			Shallow copy problem specific data attributes, deeply
			copy instance specific data attributes. 

			Time complexity: O(n)
		"""
		new_instance = copy.copy(origin)
		new_instance.solution = np.copy(origin.solution)
		if new_instance.isLowerBoundSet():
			new_instance.heur_solution = np.copy(origin.heur_solution)
		return new_instance

	def includeItem(self, item, index):
		""" Add item to the solution. 

			item ... index of the item w.r.t. the values and weights array
			index ... index of the item in the sorted vw_frac array
			
			Return true if item was sucessfully added (i.e. knapsack
			had enough capacity left). Update instance specific
			data attributes. 
		"""
		if self.curr_weight + self.weights[item] > self.max_capacity:
			return False
		self.solution[item] = 1
		self.index = index + 1
		self.curr_value += self.values[item]
		self.curr_weight += self.weights[item]
		self.heur_solution = None
		self.heur_weight = None
		self.lower_bound = None
		self.upper_bound = None
		return True

	def isMax(self):
		""" Return True if optimization problem is a maximization problem. """
		return True

	def calcUpperBound(self):
		""" Calculate upper bound value.
			
			Greedily add items into knapsack based on best value/weight 
			fractions of the items. First item which cannot be added to
			the knapsack is fractionally added to it based on the rest
			capacity of the knapsack. Upper bound rounded down if 
			specific Knapsack problem is set to be an integer problem.

			Time Complexity: O(n)
		"""
		self.upper_bound = self.curr_value
		weight = self.curr_weight
		for item in self.vw_frac_sortI[self.index:]:
			p_weight = weight + self.weights[item]
			if p_weight <= self.max_capacity:
				weight = p_weight
				self.upper_bound += self.values[item]
			else:
				self.upper_bound += (self.max_capacity - weight) \
									* self.vw_frac[item]
				break
		if self.int_problem:
			self.upper_bound = int(self.upper_bound)
		
	def calcLowerBound(self):
		""" Calculate lower bound value and set heuristic solution.
			
			Greedily add items into knapsack based on best value/weight 
			fractions of the items, traverses all possible items.

			Time Complexity: O(n) 
		"""
		self.lower_bound = self.curr_value
		self.heur_solution = np.copy(self.solution)
		self.heur_weight = self.curr_weight
		for item in self.vw_frac_sortI[self.index:]:
			p_weight = self.heur_weight + self.weights[item]
			if p_weight <= self.max_capacity:
				self.heur_weight = p_weight
				self.heur_solution[item] = 1
				self.lower_bound += self.values[item]

	def getUpperBound(self):
		""" Return upper bound value. """
		return self.upper_bound

	def getLowerBound(self):
		""" Return lower bound value. """
		return self.lower_bound

	def isUpperBoundSet(self):
		if self.upper_bound is None:
			return False
		return True

	def isLowerBoundSet(self):
		if self.lower_bound is None:
			return False
		return True

	def genInitialSolution(self):
		self.calcLowerBound()

	def getHeuristicSolution(self):
		""" Return greedy solution corresbonding to the lower bound. """
		return self.heur_solution

	def branch(self):
		""" Branch knapsack instance into subinstances. 
			
			Returns list of subinstances where in each case one distinct
			item was added to the solution. Return in reverse order
			to allow for left depth first instead of right depth first
			exploration if search strategy is set to depth first.

			Time Complexity: O(n^2) (dominant)
		"""
		subinstance_l = []
		for index in range(self.n_items-1, self.index-1, -1):
			item = self.vw_frac_sortI[index]
			subinstance = Knapsack.copy(self)
			if(subinstance.includeItem(item, index)):
				subinstance_l.append(subinstance)
		return subinstance_l
