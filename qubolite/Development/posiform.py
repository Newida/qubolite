import numpy as np
from qubolite import qubo
import networkx as nx


class posiform(qubo):
    def __init__(self, q:qubo):
        self.matrix, self.a_0 = self.convertQUBO_to_positform(q)

        self.n = 2 * q.n

    def convertQUBO_to_positform(self, q: qubo):
        #create posiform in such a way that it can directly be used for the implication network
        #therefore we add nodes x_0, not x_0 to get rid of the linear terms as described on page 5
        n = q.n
        Q = np.copy(q.m)
        #Step 1: make quadratic terms positive
        linear = np.zeros(2*n)
        linear[:n] = np.diag(q.m)
        np.fill_diagonal(Q, 0)
        quadratic = np.zeros((2*n + 2,2*n + 2))
        positive_terms = np.multiply(Q, Q > 0)
        negative_terms = np.multiply(Q, Q < 0)
        #create matrix
        quadratic[1:n+1, 1:n+1] = positive_terms
        #quadratic[n+2:, n+2:] = positive_terms
        #print(quadratic)
        quadratic[1:n+1, n+2:] = -negative_terms
        #quadratic[1:n+1, n+2:] += -negative_terms.T
        #print(quadratic)
        linear[:n] += np.sum(negative_terms, axis = 1)
        #Step 2: make linear terms positive and add to x_0 and not x_0
        linear[n:] = np.where(linear[:n] < 0, -linear[:n], 0)
        a_0 = np.sum(linear[n:])
        linear[:n][linear[:n] < 0] = 0
        quadratic[0, 1:n+1] = linear[:n]
        #quadratic[n+2:, n+2] = linear[:n]
        #print(quadratic)
        quadratic[0, n+2:] = linear[n:]
        #quadratic[1:n+1, n+1] = linear[n:]
        #print(quadratic)
        return quadratic, a_0
        

    def convert_to_implicationnetwork(self):
        pass

    def convert_toQUBO(self):
        pass
       
Q = qubo(np.array([[2, -1, 2, -2, 2], [0, 1, 1, -1, -1], [0, 0, -2, 2, -2], [0, 0, 0, -1, 2], [0, 0, 0, 0, 1]]))
p = posiform(Q)
print(p.matrix)
#p.convert_to_implicationnetwork()