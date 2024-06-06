import numpy as np
from qubolite import qubo
from igraph import Graph
import random

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
        if not G.are_connected(Q.n + (i+2), j):
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

Q = qubo(np.array([[ 4.78877637e-02, -2.06776208e-01,  7.48464713e-01,
         6.55431147e-01,  9.41043323e-01,  9.51226678e-01,
         2.28204736e-01,  1.27661879e-01,  8.12512006e-01,
        -6.11564525e-04],
       [ 0.00000000e+00,  3.72718986e-01,  3.32347713e-01,
         7.33332076e-01, -1.83492780e-01, -8.00490070e-01,
        -5.93054965e-01,  1.63875632e-01,  3.83462190e-01,
         7.23576175e-01],
       [ 0.00000000e+00,  0.00000000e+00,  2.86827017e-01,
         2.66351753e-01,  3.73514406e-01, -4.08013709e-01,
        -8.23106677e-01, -2.27533743e-01,  5.40523062e-01,
        -8.89548206e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -9.88591972e-02, -9.39609669e-01, -6.34360610e-04,
         7.01847540e-01, -4.42445385e-01, -9.11121049e-01,
         4.77044535e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00, -3.39810448e-01,  4.43384325e-01,
         3.35598430e-01,  1.51186422e-02,  7.12103273e-01,
         8.35847488e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  2.94503336e-01,
        -8.10559812e-01,  3.76861395e-01, -8.45686574e-01,
        -3.53207370e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -7.67539841e-01, -9.85784809e-01, -5.25964773e-01,
        -1.89602583e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  7.56396513e-01, -3.97028888e-01,
         5.44034664e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  9.54631024e-01,
         4.95747072e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.03966746e-01]]))
print("Input:", Q)
P, const = Q.to_posiform()
G = to_flow_graph(P)
#apply change
i, j = 4, 5
Q.m[i,j] += 0.7781629691754302
newP, c_truth = Q.to_posiform()
newG_truth = to_flow_graph(newP)
G_adapted, c_adapted = adapt_graph(Q, G, (i,j))
#d = np.linalg.norm(np.array(newG_truth.es["capacity"]) - np.array(G.es["capacity"]))
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
    for i in range(num):
        Q = qubo.random(n=n, distr='uniform', low=-1, high=1)
        P, const = Q.to_posiform()
        G = to_flow_graph(P)
        change_indices = np.random.randint(0, Q.n, 2)
        i = change_indices[0]
        j = change_indices[1]
        if i > j:
            j, i = i, j
        change = random.uniform(-1, 1)
        Q.m[i,j] += change
        newP, c_truth = Q.to_posiform()
        G_truth = to_flow_graph(newP)
        G_adapted, c_adapted = adapt_graph(Q, G, (i,j))
        diff = compare_graphs(G_adapted, G_truth)
        diff += c_truth - c_adapted
        if not np.isclose(diff, 0):
            print("+"*10)
            print(i)
            print(Q)
            print(i, j)
            print(change)
            print("Diff:", diff)
            break

test_adapt_graph(10)