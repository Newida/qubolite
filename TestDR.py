
import numpy as np
from tqdm import tqdm
from qubolite import qubo
from qubolite.preprocessing import reduce_dynamic_range
from qubolite._heuristics import MatrixOrder
import time
import cProfile

"""
a = np.arange(3**2).reshape(3,3) + 1
Q = qubo(a)
print(Q)
Q_reduced = reduce_dynamic_range(Q, heuristic='greedy0', decision='heuristic', iterations=100)
print(Q_reduced)
"""
def brute_force_solutions(Q):
    total_vectors = 2 ** Q.n
    # Generate an array of integers from 0 to 2^n - 1
    int_array = np.arange(total_vectors)
    # Use bit manipulation to get the binary representation
    binary_vectors = ((int_array[:, None] & (1 << np.arange(Q.n))) > 0).astype(int)
    energies = Q(binary_vectors)
    minima_idx = np.argmin(energies)
    return binary_vectors[minima_idx], energies[minima_idx]

def test_reduce_dynamic_range(maxDim, testSize=50):
    for n in range(2, maxDim+1):
       print("Testing for n = ", n)
       for i in tqdm(range(testSize)):
        Q = qubo.random(n=n, distr='uniform', low=-1, high=1)
        solutions_truth, _ = brute_force_solutions(Q)
        try:
            Q_reduced = reduce_dynamic_range(Q, heuristic='greedy0', decision='heuristic', iterations=100)
        except Exception as e:  
            print("Error: ", e)
            print("Q: ", Q)
            return Q, Q_reduced
        solutions_reduced, _ = brute_force_solutions(Q_reduced)
        if not np.isclose(np.linalg.norm(solutions_truth - solutions_reduced), 0):
            print("Error: Solutions are not close")
            print("Q: ", Q)
            return Q, Q_reduced

#Q, Q_reduced = test_reduce_dynamic_range(5)

#TODO:
#The following instance shows some error in the upper bound calculation cumulating over many iterations
#Investigate the error and fix it

Q = qubo(np.array([[-0.25475524,  0.1733875],
       [ 0.        , -0.21747556]]))

Q_reduced = reduce_dynamic_range(Q, heuristic='greedy0', decision='heuristic', iterations=100)
print("Q_reduced: ", Q_reduced)
truth, _ = brute_force_solutions(Q)
reduced, _ = brute_force_solutions(Q_reduced)
print("Truth: ", truth)
print("Reduced: ", reduced)

"""
Q = qubo.random(n=16, distr='uniform', low=-0.5, high=0.5)
#np.save("Benchmark.npy", Q.m)
pr = cProfile.Profile()
Q = qubo(np.load("Benchmark19to11.npy"))
print("Problem size:", Q.m.shape)
dr = MatrixOrder(Q.m).dynamic_range
print("Start DR: ", dr)
random_state = 0
#random_state = 4 creates an error
#should have stopped before comming to this since self.distances[self.distances > self.min_distance] is empty, which creates the error
np.random.seed(random_state)
start = time.time()
pr.run('Q_reduced = reduce_dynamic_range(Q, heuristic=\'greedy0\', decision=\'heuristic\', iterations=1000, random_state=random_state, upper_bound_kwargs={"restarts": 1})')
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
"""
pr = cProfile.Profile()
Q = qubo.random(n=16, distr='uniform', low=-0.5, high=0.5)
#why does the algorithm interrupt early?
#Q = qubo(np.load("Benchmarkinput.npy"))
print("Problem size:", Q.m.shape)
random_state = 0
#random_state = 4 creates an error
#should have stopped before comming to this since self.distances[self.distances > self.min_distance] is empty, which creates the error
np.random.seed(random_state)
start = time.time()
pr.run('Q_reduced, counter = reduce_dynamic_range(Q, heuristic=\'greedy0\', decision=\'heuristic\', iterations=1000, random_state=random_state, upper_bound_kwargs={"restarts": 1})')
end = time.time()
dr = MatrixOrder(Q_reduced.m).dynamic_range
t = end - start
print("DR:", dr)
print("This took " + str(t) + " seconds.")
pr.dump_stats("stats_greed0_restarts1_time_" + str(t) +  "_DR_" + str(dr) + "_counter_" + str(counter) + ".prof")
"""