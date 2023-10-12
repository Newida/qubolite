import numpy as np
from qubolite import qubo
import igraph as ig

class posiform(qubo):
    def __init__(self, q:qubo):
        self.matrix, self.a_0 = self.convertQUBO_to_positform(q)
        self.n = 2 * q.n

    def convertQUBO_to_positform(self, q: qubo):
        n = q.n
        Q = np.copy(q.m)
        #Step 1: make quadratic terms positive
        linear = np.zeros(2*n)
        linear[:n] = np.diag(q.m)
        np.fill_diagonal(Q, 0)
        quadratic = np.ones((2*n,2*n))
        quadratic[:n, :n] = np.multiply(Q, Q > 0)
        quadratic[:n, n:] = -np.multiply(Q, Q < 0)
        a = np.sum(np.multiply(Q, Q < 0), axis = 1)
        linear[:n] += np.sum(np.multiply(Q, Q < 0), axis = 1)
        #Step 2: make linear terms positive
        a_0 = 0
        linear[n:] = -linear[:n][linear[:n] < 0]
        a_0 = np.sum(linear[:n][linear[:n] < 0])
        linear[:n][linear[:n] < 0] = 0
        return quadratic, a_0
        

    def convert_to_implicationnetwork(self):
        linear = np.diag(self.matrix)
        #create adjacency matrix from posiform
        print(p.matrix)
        adjacency_matrix = np.copy((self.matrix  + self.matrix.T) / 2)
        print(adjacency_matrix)


    def convert_toQUBO(self):
        pass
       
Q = qubo(np.array([[-1,-2],[0,-4]]))
p = posiform(Q)
#p.convert_to_implicationnetwork()