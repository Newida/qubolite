
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
    return True, True

Q, Q_reduced = test_reduce_dynamic_range(10)

#TODO:
#The following instance shows some error in the upper bound calculation cumulating over many iterations
#Investigate the error and fix it
"""
Q = qubo(np.array([[-0.25633977, -0.18576958, -0.66119149,  0.98867377, -0.53236569,
         0.68374399, -0.80342775, -0.11885048,  0.94143421, -0.4059954 ],
       [ 0.        , -0.98238707, -0.99263788,  0.53172209, -0.00668407,
        -0.00323812,  0.48563132, -0.99589129, -0.75374794, -0.52181102],
       [ 0.        ,  0.        , -0.4556743 , -0.09413397,  0.65762791,
         0.03733414,  0.15233432, -0.92665415,  0.49213117, -0.27882238],
       [ 0.        ,  0.        ,  0.        ,  0.80042038,  0.8440991 ,
         0.51595295, -0.78271377, -0.26195185, -0.87610849, -0.12869783],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.43233602,
        -0.93087154, -0.23815535,  0.58743986, -0.78529219, -0.15019155],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        -0.99717483,  0.41752004, -0.69737745, -0.01257484,  0.80471809],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.87895818, -0.14830485,  0.34432571,  0.84163422],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        , -0.45234077,  0.75042841,  0.31186503],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        , -0.17975075, -0.61274694],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.06213254]]))

Q_reduced = reduce_dynamic_range(Q, heuristic='greedy0', decision='heuristic', iterations=100, random_state=4)
print("Q_reduced: ", Q_reduced)
truth, _ = brute_force_solutions(Q)
reduced, _ = brute_force_solutions(Q_reduced)
print("Truth: ", truth)
print("Reduced: ", reduced)
print("Diff:", np.linalg.norm(truth - reduced))

def find_differences(a_orig, a_mine):
    find_differences = list()
    for i in range(a_orig.shape[0]):
        for j in range(a_orig.shape[1]):
            d = a_orig[i,j] - a_mine[i,j]
            if not np.isclose(d, 0):
                find_differences.append((i,j))
    return find_differences
a_mine = np.array([[-0.25633977, -0.18576958, -0.66119149,  0.        , -0.53236569,
         0.68374399, -0.80342775, -0.11885048,  0.        , -0.4059954 ],
       [ 0.        , -0.10982629,  0.        ,  0.53172209, -0.00668407,
        -0.00323812,  0.48563132, -0.09413397, -0.75374794,  0.        ],
       [ 0.        ,  0.        , -0.4556743 , -0.09413397,  0.65762791,
         0.03733414,  0.15233432, -0.92665415,  0.49213117, -0.27882238],
       [ 0.        ,  0.        ,  0.        ,  0.80042038,  0.75366653,
         0.51595295, -0.78271377, -0.26195185, -0.87610849, -0.12869783],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.43233602,
        -0.93087154, -0.23815535,  0.58743986, -0.64297288, -0.15019155],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.41752004, -0.69737745, -0.01257484,  0.80471809],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.34432571,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        , -0.45234077,  0.        ,  0.31186503],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        , -0.17975075, -0.61274694],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.06213254]])

a_orig = np.array([[-0.25633977, -0.18576958, -0.66119149,  0.        , -0.53236569,
         0.68374399, -0.80342775, -0.11885048,  0.        , -0.4059954 ],
       [ 0.        , -0.10982629,  0.        ,  0.53172209, -0.00668407,
        -0.00323812,  0.48563132, -0.09413397, -0.75374794, -0.52181102],
       [ 0.        ,  0.        , -0.4556743 ,  0.        ,  0.65762791,
         0.03733414,  0.15233432, -0.92665415,  0.49213117, -0.27882238],
       [ 0.        ,  0.        ,  0.        ,  0.80042038,  0.75366653,
         0.51595295, -0.78271377, -0.26195185, -0.87610849, -0.12869783],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.43233602,
        -0.93087154, -0.23815535,  0.58743986, -0.64297288, -0.15019155],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.41752004, -0.69737745, -0.01257484,  0.80471809],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.34432571,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        , -0.45234077,  0.        ,  0.31186503],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        , -0.17975075, -0.61274694],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.06213254]])

#print(find_differences(a_orig, a_mine))
"""
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