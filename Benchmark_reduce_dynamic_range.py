import numpy as np
from tqdm import tqdm
from qubolite import qubo
from qubolite.preprocessing import reduce_dynamic_range
from qubolite._heuristics import MatrixOrder
import time
import cProfile
import io
import pstats

"""
Q = qubo.random(n=30, distr='uniform', low=-0.5, high=0.5)
#np.save("Benchmark.npy", Q.m)
pr = cProfile.Profile()
#Q = qubo(np.load("Benchmark19to11.npy"))
print("Problem size:", Q.m.shape)
dr = MatrixOrder(Q.m).dynamic_range
print("Start DR: ", dr)
random_state = 0
#random_state = 4 creates an error
#should have stopped before comming to this since self.distances[self.distances > self.min_distance] is empty, which creates the error
np.random.seed(random_state)
start = time.time()
pr.run('Q_reduced = reduce_dynamic_range(Q, heuristic=\'greedy0\', decision=\'heuristic\', iterations=1000, random_state=random_state)')
end = time.time()
print(Q_reduced)
dr = MatrixOrder(Q_reduced.m).dynamic_range
t = end - start
print("DR:", dr)
print("This took " + str(t) + " seconds.")

#Benchmarking given the experiments from the paper
#n = {4, 8, 12, 16}
#iterations = 1000
#values q_{ij} \in [-0.5, 0.5]
#1000 different matrices
"""

pr = cProfile.Profile()
Q = qubo.random(n=20, distr='uniform', low=-0.5, high=0.5)
#np.save("Benchmark3.npy", Q.m)
Q = qubo(np.load("Benchmark3.npy"))
print("Problem size:", Q.m.shape)
random_state = 0
#random_state = 4 creates an error
#should have stopped before comming to this since self.distances[self.distances > self.min_distance] is empty, which creates the error
np.random.seed(random_state)
start = time.time()
pr.run('Q_reduced = reduce_dynamic_range(Q, heuristic=\'greedy0\', decision=\'heuristic\', iterations=1000, random_state=random_state)')
end = time.time()
dr = MatrixOrder(Q_reduced.m).dynamic_range
t = end - start
print("DR:", dr)
print("This took " + str(t) + " seconds.")
pr.dump_stats("stats_greed0_restarts1_time_" + str(t) +  "_DR_" + str(dr))
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())
