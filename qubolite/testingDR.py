import numpy as np
from qubolite import qubo
from qubolite.preprocessing import reduce_dynamic_range
from _heuristics import MatrixOrder
import time
a = np.arange(3**2).reshape(3,3) + 1
Q = qubo(a)
Q = reduce_dynamic_range(Q, heuristic='greedy0', decision='heuristic', iterations=100)
#print(Q)

import cProfile
#Benchmarking given the experiments from the paper
#n = {4, 8, 12, 16}
#iterations = 1000
#values q_{ij} \in [-0.5, 0.5]
#1000 different matrices
pr = cProfile.Profile()
#Q = qubo.random(n=16, distr='uniform', low=-0.5, high=0.5)
#why does the algorithm interrupt early?
Q = qubo(np.load("Benchmarkinput.npy"))
print("Problem size:", Q.m.shape)
random_state = 0
#random_state = 4 creates an error
#should have stopped before comming to this since self.distances[self.distances > self.min_distance] is empty, which creates the error
np.random.seed(random_state)
start = time.time()
#pr.run('Q_reduced, counter = reduce_dynamic_range(Q, heuristic=\'greedy0\', decision=\'heuristic\', iterations=1000, random_state=random_state, upper_bound_kwargs={"restarts": 1})')
end = time.time()
#dr = MatrixOrder(Q_reduced.m).dynamic_range
#t = end - start
#print("DR:", dr)
#print("This took " + str(t) + " seconds.")
#pr.dump_stats("stats_greed0_restarts1_time_" + str(t) +  "_DR_" + str(dr) + "_counter_" + str(counter) + ".prof")