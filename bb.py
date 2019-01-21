#######################################################################
# Author: Lukas Gosch
# Date: 21.1.2019
# Description:
# 	This is a general purpose branch and bound API which supports
#	three search strategies: best-first, depth-first, breath-first.
#
#	To use this API to solve a certain problem using B&B, a class
#	representing both the problem to solve as well as an instance of 
#	this problem has to be created. This class must inherit from the 
#	Instance class and overwrite it's "abstract" functions such as  
#	lower and upper bound calculation and branching. Given a user has 
#	implemented a class Problem inheriting from Instance and has 
#	initialized an object obj of type Problem. She can then drive
#	the optimization through calling obj.solve().
#
#	The API can be used to solve min- and maximization problems. If
#	a certain problem instance is a minimization or maximization 
#	problem has to be told the API through overwriting the function 
#	isMax().
#
#	An example implementation for the 0/1 Knapsack problem is provided 
#	through a class Knapsack inheriting from Instance, see Knapsack.py 
#	
#	A custom search strategy can be added by modifying the methods 
#	in the InstanceSet class.
#######################################################################

import sys
import importlib
import argparse
import queue
import numpy as np

class Instance:
	""" Abstract Problem Instance Class, if one wants to add a Problem
		Class to solve using B&B, it must inherit from Instance.
	"""

	def isMax(self):
		""" Return True if optimization problem is a maximization problem. """
		raise NotImplementedError("isMax implementation missing.")

	def calcUpperBound(self):
		""" Calculate upper bound value. 
			
			If the problem is a minimization problem, set valid heuristic
			solution.
		"""
		raise NotImplementedError("calcUpperBound implementation missing.")

	def calcLowerBound(self):
		""" Calculate lower bound value. 
			
			If the problem is a maximiztion problem, set valid heuristic
			solution.
		"""
		raise NotImplementedError("calcLowerBound implementation missing.")

	def getUpperBound(self):
		""" Return upper bound value. """
		raise NotImplementedError("getUpperBound implementation missing.")

	def getLowerBound(self):
		""" Return lower bound value. """
		raise NotImplementedError("getLowerBound implementation missing.")

	def isUpperBoundSet(self):
		""" Return true if an up-to-date upper bound has been calculated. """
		raise NotImplementedError("isUpperBoundSet implementation missing.")

	def isLowerBoundSet(self):
		""" Return true if an up-to-date lower bound has been calculated. """
		raise NotImplementedError("isLowerBoundSet implementation missing.")

	def genInitialSolution(self):
		""" Generate an initial heuristic solution. """
		raise NotImplementedError("genInitialSolution implementation missing.")

	def getHeuristicSolution(self):
		""" Return a valid heuristic solution. 

			The returned solution should correspond to the lower bound
			(upper bound) value in the maximization (minimization) problem.
			The solve() methods ensures that this function is only called
			if isLowerBoundSet() (isUpperBoundSet()) returns true.
		"""
		raise NotImplementedError("getHeuristicSolution implementation missing.")

	def getHeuristicSolutionValue(self):
		""" Return value of the heuristic solution. """
		if self.isMax():
			return self.getLowerBound()
		return self.getUpperBound()

	def branch(self):
		""" Generate and return iterator over subinstances. """
		raise NotImplementedError("branch implementation missing.")

	def solve(self, search_strategy = 'best_first', max_branches = -1):
		""" Solve problem using branch & bound.

			search_strategy ... specify a ceratin search strategy for
								the branch & bound
			max_branches	...	specify a maximum number of branches
								generated before the algorithm should
								stop 
			
			Returns the found solution, its value, the number of branches
			generated and the number of investigated instances.
		"""
		# Count number of branches and iterations
		branch_c = 0
		iter_c = 0

		# Initialize best solution as initial solution
		self.genInitialSolution()
		best_I = self

		I_set = InstanceSet(self, search_strategy)
		# Evaluate promising instances until no promising instance left
		# or (if set) maximum number of branch-processes reached.
		while (not I_set.isEmpty()
			   and (branch_c < max_branches or max_branches == -1)):
			# Remove next instance from instance set
			I = I_set.getNextInstance()
			# Look at instance if it could contain a better global solution
			if I.isPromising(best_I):
				if I.isBetterThen(best_I):
					best_I = I
				# Branch current instance if still promising
				if I.isPromising(best_I):
					for sub_I in I.branch():
						# Add subinstance to instance set if promising
						if sub_I.isPromising(best_I):
							I_set.addInstance(sub_I)
					branch_c += 1
			iter_c += 1

		opt_solution = best_I.getHeuristicSolution()
		opt_value = best_I.getHeuristicSolutionValue()
		return opt_solution, opt_value, branch_c, iter_c

	def isPromising(self, current):
		""" Return True if instance self could contain a better
			solution then the heuristic solution set in the 
			current instance.
		"""
		if self.isMax():
			if not self.isUpperBoundSet():
				self.calcUpperBound()
			if self.getUpperBound() > current.getLowerBound():
				return True
		else:
			if not self.isLowerBoundSet():
				self.calcLowerBound()
			if self.getLowerBound() < current.getUpperBound():
				return True
		return False

	def isBetterThen(self, current):
		""" Return True if self contains a better heuristic solution
			then current.
		"""
		if self.isMax():
			if not self.isLowerBoundSet():
				self.calcLowerBound()
			return self.getLowerBound() > current.getLowerBound()
		else:
			if not self.isUpperBoundSet():
				self.calcUpperBound()
			return self.getUpperBound() < current.getUpperBound()

	def __lt__(self, other):
		""" Return True if instance self holds potentially
			better solutions then instance other. 
		"""
		if self.isMax():
			return self.getUpperBound() > other.getUpperBound()
		else:
			return self.getLowerBound() < other.getLowerBound()

class InstanceSet:
	""" Holds a set of Instances of a certain problem. """

	def __init__(self, instance, search_strategy):
		""" Initializes a set of instances for a certain optimization problem.
			
			The data structure used to store subinstances of a problem depends
			on the chosen search_strategy.

			search_strategies: 
			-'best_first': priority queue is used, provides fast adding
						   and removing of instances from the instance
						   set (O(log n)). Definition of "best" depends
						   on maximization or minimization problem.
			-'depth_first': O(1) adding and removing
			-'breath_first': O(1) adding and removing
		"""
		self._sstrategy = search_strategy
		if self._sstrategy == 'best_first':
			self._iset = queue.PriorityQueue()
			# Check if bound for sorting is set
			if instance.isMax():
				if not instance.isUpperBoundSet():
					instance.calcUpperBound()
			else:
				if not instance.isLowerBoundSet():
					instance.calcLowerBound()
			self.addInstance(instance)
		elif self._sstrategy == 'depth_first':
			self._iset = queue.LifoQueue()
			self.addInstance(instance)
		elif self._sstrategy == 'breath_first':
			self._iset = queue.Queue()
			self.addInstance(instance)
		else:
			sys.exit("Search strategy not supported by InstanceSet.")

	def isEmpty(self):
		""" Return true if no problem instances are in the set. """
		if self._sstrategy == 'best_first':
			return self._iset.empty()
		elif self._sstrategy == 'depth_first':
			return self._iset.empty()
		elif self._sstrategy == 'breath_first':
			return self._iset.empty()

	def getNextInstance(self):
		""" Remove and return the next instance in the instance set. 
			Time complexity: - 'best_first': O(log n)
							 - 'depth_first': O(1)
							 - 'breath_first': O(1)
		"""
		if self._sstrategy == 'best_first':
			return self._iset.get()
		elif self._sstrategy == 'depth_first':
			return self._iset.get()
		elif self._sstrategy == 'breath_first':
			return self._iset.get()

	def addInstance(self, instance):
		""" Add instance to the instance set. 
			Time complexity: - 'best_first': O(log n)
							 - 'depth_first': O(1)
							 - 'breath_first': O(1)
		"""
		if self._sstrategy == 'best_first':
			self._iset.put(instance)
		elif self._sstrategy == 'depth_first':
			self._iset.put(instance)
		elif self._sstrategy == 'breath_first':
			self._iset.put(instance)