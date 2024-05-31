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
    #this function assume that i < j
    P, c = Q.to_posiform()
    i, j = change_idx

    #x_i => edge x_0 not_x_i and x_i not_x_0
    if not G.are_connected(0, Q.n + (i+2)):
        G.add_edge(0, Q.n + (i+2))
        G.add_edge((i+1), Q.n + 1)
    G.es[G.get_eid(0, Q.n + (i+2))]["capacity"] = P[0, i,i]/2.0
    G.es[G.get_eid(i+1, Q.n + 1)]["capacity"] = P[0, i,i]/2.0

    #not_x_i => edge x_0 x_i and not_x_i not_x_0
    if not G.are_connected(0, i+1):
        G.add_edge(0, (i+1))
        G.add_edge(Q.n + (i+2), Q.n + 1)
    G.es[G.get_eid(0, i+1)]["capacity"] = P[1, i,i]/2.0
    G.es[G.get_eid(Q.n + (i+2), Q.n + 1)]["capacity"] = P[1, i,i]/2.0

    #x_i x_j => edge x_i not_x_j and x_j not_x_i
    if not G.are_connected(i+1, Q.n + (j+2)):
        G.add_edge((i+1), Q.n + (j+2))
        G.add_edge((j+1), Q.n + (i+2))    
    G.es[G.get_eid(i+1, Q.n + (j+2))]["capacity"] = P[0, i,j]/2.0
    G.es[G.get_eid(j+1, Q.n + (i+2))]["capacity"] = P[0, i,j]/2.0

    #x_i not_x_j => edge x_i x_j and not_x_j not_x_i
    if not G.are_connected(i+1, j+1):
        G.add_edge((i+1), (j+1))
        G.add_edge(Q.n + (j+2), Q.n + (i+2))
    G.es[G.get_eid(i+1, j+1)]["capacity"] = P[1, i,j]/2.0
    G.es[G.get_eid(Q.n + (j+2), Q.n + (i+2))]["capacity"] = P[1, i,j]/2.0
    
    #not_x_i x_j => edge not_x_i not_x_j and x_j x_i
    if not G.are_connected(Q.n + (i+2), Q.n + (j+2)):
        G.add_edge(Q.n + (i+2), Q.n + (j+2))
        G.add_edge((j+1), (i+1))
    G.es[G.get_eid(Q.n + (i+2), Q.n + (j+2))]["capacity"] = P[1, i,j]/2.0
    G.es[G.get_eid((j+1), (i+1))]["capacity"] = P[1, i,j]/2.0


    #not_x_i not_x_j => edge not_x_i x_j and x_j not_x_i
    if not G.are_connected(Q.n + (i+2), j):
        G.add_edge(Q.n + (i+2), j)
        G.add_edge(Q.n + (j+2), i)
    G.es[G.get_eid(Q.n + (i+2), j)]["capacity"] = P[1, i,j]/2.0
    G.es[G.get_eid(Q.n + (j+2), i)]["capacity"] = P[1, i,j]/2.0

    return c


#Q = qubo(np.array([[ 0.05755527, -0.88476015, -0.78392966],
#       [ 0.        , -0.38025914, -0.64084856],
#       [ 0.        ,  0.        , -0.5782138 ]]))
Q = qubo.random(n=3, distr='uniform', low=-1, high=1)
print("Input:", Q)
P, const = Q.to_posiform()
G = to_flow_graph(P)
#apply change
i, j = 0, 2
Q.m[i,j] -= 0.5
newP, newconst = Q.to_posiform()
newC = adapt_graph(Q, G, (i,j))
newG_truth = to_flow_graph(newP)
d = np.linalg.norm(np.array(newG_truth.es["capacity"]) - np.array(G.es["capacity"]))
if not np.isclose(d, 0):
    print("Original Posiform:", P)
    print("Original graph:")
    for e in G.es:
        print(e.source, e.target, e["capacity"])
    print("+"*50)
    print("Changed Q:", Q)
    print("Changed Posiform:", newP)
    print("Changed graph:")
    for edge in newG_truth.es:
         print(edge.source, edge.target, edge["capacity"])
    print("+"*50)
    print("my graph:")
    for edge in G.es:
        print(edge.source, edge.target, edge["capacity"])
print("Diff:", d)