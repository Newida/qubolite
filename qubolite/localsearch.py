import numpy as np
from qubolite import qubo
import numpy.random as rng
n = 3
x = rng.random(n) < 0.5
x_ = x.copy()
Q = qubo(1 + np.arange(n**2).reshape(n,n))
print(Q)

m_  = np.triu(Q.m, 1)
m_ += m_.T
print("M_", m_)
sign = 1-2*x
result = sign*(np.diag(Q.m)+(m_ @ x))
print("result2:", result)

m2 = (Q.m + Q.m.T)
np.fill_diagonal(m2, 0)
print("M_wrong", m2)
sign2 = 1 - 2*x
result2 = sign * (np.diag(Q.m) + m2 @ x)
print("result_wrong:", result2)