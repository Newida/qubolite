import numpy as np
from qubolite import qubo
from igraph import Graph
import random
from tqdm import tqdm

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

def adapt_graph(Q, G_orig, change_idx):
    G = G_orig.copy()
    #this function assume that i < j
    P, c = Q.to_posiform()
    i, j = change_idx
    #x_i => edge x_0 not_x_i and x_i not_x_0
    if not G.are_connected(0, Q.n + (i+2)):
        if P[0, i,i] != 0:
            G.add_edge(0, Q.n + (i+2), capacity=P[0, i,i]/2.0)
            G.add_edge((i+1), Q.n + 1, capacity=P[0, i,i]/2.0)
    else:
        G.es[G.get_eid(0, Q.n + (i+2))]["capacity"] = P[0, i,i]/2.0
        G.es[G.get_eid(i+1, Q.n + 1)]["capacity"] = P[0, i,i]/2.0

    #not_x_i => edge x_0 x_i and not_x_i not_x_0
    if not G.are_connected(0, i+1):
        if P[1, i,i]/2.0 != 0:
            G.add_edge(0, (i+1), capacity=P[1, i,i]/2.0)
            G.add_edge(Q.n + (i+2), Q.n + 1, capacity=P[1, i,i]/2.0)
    else:
        G.es[G.get_eid(0, i+1)]["capacity"] = P[1, i,i]/2.0
        G.es[G.get_eid(Q.n + (i+2), Q.n + 1)]["capacity"] = P[1, i,i]/2.0
    if i != j:
        #x_i x_j => edge x_i not_x_j and x_j not_x_i
        if not G.are_connected(i+1, Q.n + (j+2)):
            if P[0, i,j]/2.0 != 0:
                G.add_edge((i+1), Q.n + (j+2), capacity=P[0, i,j]/2.0)
                G.add_edge((j+1), Q.n + (i+2), capacity=P[0, i,j]/2.0)
        else:
            G.es[G.get_eid(i+1, Q.n + (j+2))]["capacity"] = P[0, i,j]/2.0
            G.es[G.get_eid(j+1, Q.n + (i+2))]["capacity"] = P[0, i,j]/2.0
    
        #x_i not_x_j => edge x_i x_j and not_x_j not_x_i
        if not G.are_connected((i+1), (j+1)):
            if P[1, i,j]/2.0 != 0:
                G.add_edge((i+1), (j+1), capacity=P[1, i,j]/2.0)
                G.add_edge(Q.n + (j+2), Q.n + (i+2), capacity=P[1, i,j]/2.0)
        else:
            G.es[G.get_eid((i+1), (j+1))]["capacity"] = P[1, i,j]/2.0
            G.es[G.get_eid(Q.n + (j+2), Q.n + (i+2))]["capacity"] = P[1, i,j]/2.0

    return G, c

def compare_graphs(G_adapted, G_truth):
    # Create a set of edges for both graphs
    edges_G1 = set((e.source, e.target) for e in G_adapted.es)
    edges_G2 = set((e.source, e.target) for e in G_truth.es)
    
    # Initialize a dictionary to store the comparison results
    diff = 0
    
    # Iterate through the edges of G_1
    for edge in edges_G1:
        source, target = edge
        capacity_G1 = G_adapted.es[G_adapted.get_eid(source, target)]['capacity']
        
        # Check if the edge exists in G_2
        if edge in edges_G2:
            capacity_G2 = G_truth.es[G_truth.get_eid(source, target)]['capacity']
        else:
            capacity_G2 = 0.0
        
        diff += capacity_G1 - capacity_G2
    
    return diff

Q = qubo(np.array([[-0.0347891 , -0.40028512, -0.68155756,  0.96315303],
       [ 0.        , -0.6173675 ,  0.87557738, -0.47181135],
       [ 0.        ,  0.        , -0.03672531, -0.99585558],
       [ 0.        ,  0.        ,  0.        , -0.69890477]]))
print("Input:", Q)
P, const = Q.to_posiform()
G = to_flow_graph(P)
i, j = 1, 3
Q.m[i,j] += 0.8781377842596512
newP, c_truth = Q.to_posiform()
newG_truth = to_flow_graph(newP)
G_adapted, c_adapted = adapt_graph(Q, G, (i,j))
d = compare_graphs(G_adapted, newG_truth)
d += c_truth - c_adapted
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
    for edge in G_adapted.es:
        print(edge.source, edge.target, edge["capacity"])
print("Diff:", d)

def test_adapt_graph(n, num=1000):
    for i in tqdm(range(num)):
        Q_orig = qubo.random(n=n, distr='uniform', low=-1, high=1)
        P, _ = Q_orig.to_posiform()
        Q = Q_orig.copy()
        G = to_flow_graph(P)
        change_indices = np.random.randint(0, Q.n, 2)
        i = change_indices[0]
        j = change_indices[1]
        if i > j:
            j, i = i, j
        change = random.uniform(-1, 1)
        Q.m[i,j] += change
        newP, c_truth = Q.to_posiform()
        newG_truth = to_flow_graph(newP)
        G_adapted, c_adapted = adapt_graph(Q, G, (i,j))
        d = compare_graphs(G_adapted, newG_truth)
        d += c_truth - c_adapted
        if not np.isclose(d, 0):
            print("+"*10)
            print(Q_orig)
            print(Q)
            print(i, j)
            print("change:", change)
            print("Diff:", d)
            return G_adapted, newG_truth

"""
G_adapted, newG_truth = test_adapt_graph(4)
print("+"*50)
print("TRUTH:")
for edge in newG_truth.es:
    print(edge.source, edge.target, edge["capacity"])
print("ADAPTED:")
for edge in G_adapted.es:
    print(edge.source, edge.target, edge["capacity"])
print("+"*50)

print(compare_graphs(G_adapted, newG_truth))
"""