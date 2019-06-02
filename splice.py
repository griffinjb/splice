import numpy as np 
from numpy import array as a 
from multiprocessing_generator import ParallelGenerator
from random import random as r
from random import gauss as g
from random import shuffle
import matplotlib.pyplot as plt
from time import sleep

# with ParallelGenerator(
# 	self.xygen(wn1a,wn2a),
# 	max_lookahead=200) as xyg:
# 	for x,y in xyg:
# 		self.model.fit(x,y,epochs=1)

class agent:
	c = a([0,0])	# Row,Col
	E = []			# Sensing Matrix
	env = ''		# Environment Generator
	f = 0			# Fitness Score

	G = []			# Gene Matrix
	P = []			# Next Step Policy


	def __init__(self,env,cfg):
		self.c = [int(g(0,10)),int(g(0,10))]
		self.env = env
		self.E = env.getSensingMatrix(self.c)
		self.G = gene(cfg)

	def step(self):

		self.E = self.env.getSensingMatrix(self.c)

		self.P = self.G.A @ self.E @ self.G.B
		for i in range(len(self.P)):
			if self.P[i] == np.array(self.P).max():
				break
		# Up
		if i == 0:
			self.c += a([1,0])
		# Down
		elif i == 1:
			self.c += a([-1,0])
		# Left
		elif i == 2:
			self.c += a([0,-1])
		# Right
		elif i == 3:
			self.c += a([0,1])
		# Up Right
		elif i == 4:
			self.c += a([1,1])
		# Down Right
		elif i == 5:
			self.c += a([-1,1])
		# Down Left
		elif i == 6:	
			self.c += a([-1,-1])
		# Up Left
		elif i == 7:
			self.c += a([1,-1])

		self.f += self.env.consume(self.c)

class genePool:

	P = []	# Gene Pool [N_A,Dir,Vision]

	def getShuffleReducePool(self,agents):
		tmp = []

		fs = []
		for agent in agents:
			fs.append(agent.f)
		idx = sorted(range(len(fs)), key=lambda k: fs[k])
		idx = idx[int(len(idx)/2):]


		for i in idx:
			tmp.append(agents[i].G.W)
		tmp = a(tmp)
		for i in range(tmp.shape[1]):
			for j in range(tmp.shape[2]):
				np.random.shuffle(tmp[:,i,j])

		self.P = tmp

	def mutate(self):
		self.P += np.random.normal(0,.1,self.P.shape)

	# def pltAvg(self):


	def reproduce(self,agents):

		self.getShuffleReducePool(agents)
		self.mutate()

		fs = []
		for agent in agents:
			fs.append(agent.f)
		idx = sorted(range(len(fs)), key=lambda k: fs[k])
		try:
			idx = idx[:int(len(idx)/2)]
		except Exception as e:
			print(e)
			print('Use Even Number of Agents')

		PCTR = 0
		for i in range(len(agents)):
			if i in idx:
				agents[i].G.W = self.P[PCTR]
				agents[i].G.A = agents[i].G.W[:7,:]
				agents[i].G.B = agents[i].G.W[8,:].T
				PCTR += 1

		return(agents)

class gene:
	A = []	# Weight Matrix Vertical
	B = []	# Weight Matrix Horizontal
	W = []	# Weigh Matrix [A B.T].T

	def __init__(self,c):
		# [up down left right UR DR DL UL]

		# P = A@E@B
		# 8X1 = 8XV VXV VX1
		self.W = np.random.normal(0,1,[9,c.V])
		self.A = self.W[:7,:]
		self.B = self.W[8,:].T

class environment:

	master = {}
	c = ''

	def __init__(self,c):
		self.c = c

	def getSensingMatrix(self,coord):
		E = np.zeros([self.c.V,self.c.V])
		for i in range(int(coord[0]-(self.c.V-1)/2),int(coord[0]+(self.c.V-1)/2+1)):
			for j in range(int(coord[1]-(self.c.V-1)/2),int(coord[1]+(self.c.V-1)/2+1)):
				ir = int(i - coord[0] + (self.c.V-1)/2)
				jr = int(j - coord[1] + (self.c.V-1)/2)
				cstr = '['+str(i)+','+str(j)+']'
				if cstr not in self.master:
					if r()*100 <= self.c.S and np.linalg.norm([coord[0],coord[1]])<self.c.B:
						self.master[cstr] = 1
						E[ir,jr] = 1
					else:
						self.master[cstr] = 0
				else:
					E[ir,jr] = self.master[cstr]

		return(E)

	def consume(self,crd):
		cstr = '['+str(crd[0])+','+str(crd[1])+']'

		if self.master[cstr]:
			self.master[cstr] = 0
			return(1)
		else:
			return(0)

	def plot(self,A):
		master = self.master

		# Get range
		xm	= 0
		ym	= 0
		xma	= 0
		yma	= 0
		for key in master:
			x = eval(key)[0]
			y = eval(key)[1]

			if x > xma:
				xma = x 
			if x < xm:
				xm = x
			if y > yma:
				yma = y 
			if y < ym:
				ym = y

		grid = np.zeros([xma-xm+1,yma-ym+1])

		for key in master:
			xr = eval(key)[0] - xm
			yr = eval(key)[1] - ym
			grid[xr,yr] = master[key]

		for ag in A:
			xr = ag.c[0]-xm 
			yr = ag.c[1]-ym 
			grid[xr,yr] = -1

		plt.clf()
		plt.imshow(grid,aspect='equal')
		plt.show(block=False)
		plt.pause(.001)

class splice:

	A = []
	c = ''
	GP = ''

	def __init__(self,c):

		self.c = c
		E = environment(c)
		self.A = [agent(E,c) for i in range(c.N_A)]
		self.GP = genePool()

	def sharedEnvironmentTrain(self):

		for j in range(300000):
			# initialize environment
			self.E = environment(c)
			self.E.master = {}
			for ag in self.A:
				ag.env = self.E
				ag.c = [int(g(0,10)),int(g(0,10))]

			# For each agent, take step
			for i in range(self.c.L):
				for a in self.A:
					a.step()
				self.E.plot(self.A)



			# Reproduce
			self.A = self.GP.reproduce(self.A)

			avg = 0
			maxv = 0
			for ag in self.A:
				avg += ag.f
				if ag.f > maxv:
					maxv = ag.f
				ag.f = 0

			avg /= self.c.N_A

			print('gen'+str(j)+': ')
			print(avg)
			print(maxv)

		plt.imshow(ag.G.W,aspect='auto')
		plt.show(block=False)

class cfg:

	N_A = 10	# Num Agents
	S 	= 20	# Sparsity per 100
	V	= 10	# Vision / Size of sensing matrix
	B	= 1000	# Resource Boundary || x,y || > 1000
	L 	= 100   # Lifespan

	def __init__(self,ID):
		if ID == 1:
			self.N_A 	= 100
			self.S 		= 99
			self.V 		= 5
			self.B 		= 40


if __name__ == '__main__':


	c = cfg(1)
	
	s = splice(c)
	s.sharedEnvironmentTrain()

	plt.show()



