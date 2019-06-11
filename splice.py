import numpy as np 
from numpy import array as a 
from multiprocessing_generator import ParallelGenerator
from random import random as r
from random import gauss as g
from random import shuffle
import matplotlib.pyplot as plt
from time import sleep
from pickle import dump, load 


class genePool:



class splice:




		# Initialize Interface

		# Init Agent Pool

		# Init Gene Pool

		# For agent

			# Test Config

		# Mutate

		# Reproduce 





class cfg:

	N_A = 10	# Num Agents
	S 	= 20	# Sparsity per 100
	V	= 10	# Vision / Size of sensing matrix
	B	= 1000	# Resource Boundary || x,y || > 1000
	L 	= 100   # Lifespan
	M 	= 10 	# Memory
	FN 	= ''	# Gene Pool Filename | Ignore if Empty
	interface = ''	# Application Interface

	def __init__(self,ID):
		
		# V5 M5
		if ID == 1:
			self.N_A 	= 40
			self.S 		= 60
			self.V 		= 5
			self.B 		= 30
			self.FN 	= 'well_trained_v5wM.p'
			self.L 		= 40
			self.M 		= 5
			import resource_acquisition as ra
			self.interface = 

if __name__ == '__main__':
	print('temp')






