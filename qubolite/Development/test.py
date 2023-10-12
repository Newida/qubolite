import numpy as np
from qubolite import qubo

class posiform(qubo):
    def __init__(self, q:qubo):
        self.quadratic, self.a_0 = self.convertQUBO_to_positform(q)
        self.n = 2 * q.n

    def convertQUBO_to_positform(self, q: qubo):
        n = q.n
        Q = np.copy(q.m)
        #Step 1: make quadratic terms positive
        linear = np.zeros(2*n)
        linear[:n] = np.diag(q.m)
        np.fill_diagonal(Q, 0)
        quadratic = np.zeros((2*n,2*n))
        quadratic[:n, :n] = np.multiply(Q, Q > 0)
        quadratic[n:, n:] = -np.multiply(Q, Q < 0)
        a = np.sum(np.multiply(Q, Q < 0), axis = 1)
        linear[:n] += np.sum(np.multiply(Q, Q < 0), axis = 1)
        #Step 2: make linear terms positive
        a_0 = 0
        linear[n:] = -linear[:n][linear[:n] < 0]
        a_0 = np.sum(linear[:n][linear[:n] < 0])
        linear[:n][linear[:n] < 0] = 0
        np.fill_diagonal(quadratic, linear)
        return quadratic, a_0
        

    def convert_to_implicationnetwork(self):
        pass

    def convert_toQUBO(self):
        pass
       
Q = qubo(np.array([[-1,-2],[0,-4]]))
p = posiform(Q)
print(p.quadratic, p.a_0, p.n)