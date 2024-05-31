import numpy as np
from qubolite import qubo
from igraph import Graph

def to_flow_graph(P):
        n = P.shape[1]
        G = Graph(directed=True)
        vertices = np.arange(n + 1)
        negated_vertices = np.arange(n + 1, 2 * n + 2)
        # all vertices for flow graph
        all_vertices = np.concatenate([vertices, negated_vertices])
        G.add_vertices(all_vertices)
        # arrays of vertices containing node x0
        n0 = np.kron(vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
        np.fill_diagonal(n0, np.zeros(n))
        nn0 = np.kron(negated_vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
        np.fill_diagonal(nn0, (n + 1) * np.ones(n))
        # arrays of vertices not containing node x0
        n1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], vertices[1:])
        nn1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], negated_vertices[1:])

        n0_nn1 = np.stack((n0, nn1), axis=-1) # edges from ni to !nj
        n1_nn0 = np.stack((n1, nn0), axis=-1) # edges from nj to !ni
        n0_n1 = np.stack((n0, n1), axis=-1) # edges from ni to nj
        nn1_nn0 = np.stack((nn1, nn0), axis=-1) # edges from !nj to !ni
        pos_indices = np.invert(np.isclose(P[0], 0))
        neg_indices = np.invert(np.isclose(P[1], 0))
        # set capacities to half of posiform parameters
        capacities = 0.5 * np.concatenate([P[0][pos_indices], P[0][pos_indices],
                                           P[1][neg_indices], P[1][neg_indices]])
        edges = np.concatenate([n0_nn1[pos_indices], n1_nn0[pos_indices],
                                n0_n1[neg_indices], nn1_nn0[neg_indices]])
        G.add_edges(edges)
        G.es["capacity"] = capacities
        return G

def adapt_graph(Q, G, change_idx):
    #it is assume that i < j
    P, c = Q.to_posiform()
    i, j = change_idx
    if G.are_connected(0, Q.n + (i+2)):
        #x_i => edge x_0 not_x_i and x_i not_x_0
        G.es[G.get_eid(0, Q.n + (i+2))]["capacity"] = P[0, i,i]/2.0
        G.es[G.get_eid(i+1, Q.n + 1)]["capacity"] = P[0, i,i]/2.0
    if G.are_connected(0, i+1):
        #not_x_i => edge x_0 x_i and not_x_i not_x_0
        G.es[G.get_eid(0, i+1)]["capacity"] = P[1, i,i]/2.0
        G.es[G.get_eid(Q.n + (i+2), Q.n + 1)]["capacity"] = P[1, i,i]/2.0
    if G.are_connected(i+1, Q.n + (j+2)):
        #x_i x_j => edge x_i not_x_j and x_j not_x_i
        G.es[G.get_eid(i+1, Q.n + (j+2))]["capacity"] = P[0, i,j]/2.0
        G.es[G.get_eid(j+1, Q.n + (i+2))]["capacity"] = P[0, i,j]/2.0
    if G.are_connected(i+1, j+1):
        #x_i not_x_j => edge x_i x_j and not_x_j not_x_i
        G.es[G.get_eid(i+1, j+1)]["capacity"] = P[1, i,j]/2.0
        G.es[G.get_eid(Q.n + (j+2), Q.n + (i+2))]["capacity"] = P[1, i,j]/2.0

    return c

#Q = qubo.random(n=3, distr='uniform', low=-1, high=1)
Q = qubo(np.array([[ 0.06498978, -0.86825328,  0.02054162],
       [ 0.        , -0.49789045, -0.12046697],
       [ 0.        ,  0.        , -0.74605798]]))
print("Input:", Q)
P, const = Q.to_posiform()
print("Posiform:", P)
print("Constant:", const)
G = to_flow_graph(P)
#apply change
Q.m[0,1] -= 0.5
print("Changed Q:", Q)
newP, newconst = Q.to_posiform()
print("Changed Posiform:", newP)
print("Changed constant:", newconst)
print("+"*50)
print("original Graph:")
for e in G.es:
    print(e.source, e.target, e["capacity"])
newC = adapt_graph(Q, G, (0,1))
newG_truth = to_flow_graph(newP)
print("+"*50)
for e in newG_truth.es:
    print(e.source, e.target, e["capacity"])
print("+"*50)
for e in G.es:
    print(e.source, e.target, e["capacity"])
print("+"*50)