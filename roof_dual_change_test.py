import numpy as np
from qubolite import qubo, assignment
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

def adapt_graph(P, G_orig, change_idx):
    """Given the old graph G_orig and the new posiform P, adapt the graph to the new posiform"""
    n = P.shape[1]
    G = G_orig.copy()
    #this function assume that i < j
    i, j = change_idx
    #x_i => edge x_0 not_x_i and x_i not_x_0
    if not G.are_connected(0, n + (i+2)):
        if P[0, i,i] != 0:
            G.add_edge(0, n + (i+2), capacity=P[0, i,i]/2.0)
            G.add_edge((i+1), n + 1, capacity=P[0, i,i]/2.0)
    else:
        G.es[G.get_eid(0, n + (i+2))]["capacity"] = P[0, i,i]/2.0
        G.es[G.get_eid(i+1, n + 1)]["capacity"] = P[0, i,i]/2.0

    #not_x_i => edge x_0 x_i and not_x_i not_x_0
    if not G.are_connected(0, i+1):
        if P[1, i,i]/2.0 != 0:
            G.add_edge(0, (i+1), capacity=P[1, i,i]/2.0)
            G.add_edge(n + (i+2), n + 1, capacity=P[1, i,i]/2.0)
    else:
        G.es[G.get_eid(0, i+1)]["capacity"] = P[1, i,i]/2.0
        G.es[G.get_eid(n + (i+2), n + 1)]["capacity"] = P[1, i,i]/2.0
    if i != j:
        #x_i x_j => edge x_i not_x_j and x_j not_x_i
        if not G.are_connected(i+1, n + (j+2)):
            if P[0, i,j]/2.0 != 0:
                G.add_edge((i+1), n + (j+2), capacity=P[0, i,j]/2.0)
                G.add_edge((j+1), n + (i+2), capacity=P[0, i,j]/2.0)
        else:
            G.es[G.get_eid(i+1, n + (j+2))]["capacity"] = P[0, i,j]/2.0
            G.es[G.get_eid(j+1, n + (i+2))]["capacity"] = P[0, i,j]/2.0
    
        #x_i not_x_j => edge x_i x_j and not_x_j not_x_i
        if not G.are_connected((i+1), (j+1)):
            if P[1, i,j]/2.0 != 0:
                G.add_edge((i+1), (j+1), capacity=P[1, i,j]/2.0)
                G.add_edge(n + (j+2), n + (i+2), capacity=P[1, i,j]/2.0)
        else:
            G.es[G.get_eid((i+1), (j+1))]["capacity"] = P[1, i,j]/2.0
            G.es[G.get_eid(n + (j+2), n + (i+2))]["capacity"] = P[1, i,j]/2.0

    return G

def clamp_graph(Q, P, G_orig, change_idx, values):
    G = G_orig.copy()
    n = Q.n
    #this function assume that i < j
    i, j = change_idx
    v_i, v_j = values
    #setting i to 0:
    #edges from x_0x_i and x_0 not_x_i
    diag_edges = [(0, n+i+2), (i+1, n+1), (0, i+1), (n+i+2, n+1)]
    #edges with source x_i
    x_i_edges = np.repeat(i + 1, 2*((n-(i+1)) + (2*n + 1) - (n+i+2))).reshape(-1, 2)
    x_i_edges[:n-(i+1), 1] = np.arange(i+2, (n)+1)
    x_i_edges[n-(i+1):, 1] = (n + 1) + np.arange(i+2, n+1)
    #add edges with target not_x_i
    not_x_i_edges = np.repeat(n + (i + 2), 2*((n-(i+1)) + (2*n + 1) - (n+i+2))).reshape(-1, 2)
    not_x_i_edges[:n-(i+1), 0] = np.arange(i+2, n+1)
    not_x_i_edges[n-(i+1):, 0] = (n + 1) + np.arange(i+2, n+1)

    #convert edges to edge_ids
    diag_edges_list = G.get_eids(diag_edges, directed=True, error=False)
    x_i_edges_list = G.get_eids(x_i_edges, directed=True, error=False)
    not_x_i_edges_list = G.get_eids(not_x_i_edges, directed=True, error=False)
    #clear edges that do not exist
    diag_edges_list = list(filter((-1).__ne__, diag_edges_list))
    x_i_edges_list = list(filter((-1).__ne__, x_i_edges_list))
    not_x_i_edges_list = list(filter((-1).__ne__, not_x_i_edges_list))
    #set edge capacities to 0
    G.es[diag_edges_list]["capacity"] = 0
    G.es[x_i_edges_list]["capacity"] = 0
    G.es[not_x_i_edges_list]["capacity"] = 0
    
    #setting j to 0:
    #diagonal elements x_0
    diag_x_0 = np.zeros((2*n+1,2))
    diag_x_0[:, 1] = np.arange(1, 2*n+2)
    #diagonal elements not_x_0
    diag_not_x_0 = np.repeat(n+1, 2*(2*n+2)-2).reshape(-1, 2)
    diag_not_x_0[:, 0] = np.arange(1, 2*n+2)
    #TODO: non-diagonal elements missing
    #TODO: use clamp posiform to calculate the new values
    
    #convert edges to edge_ids
    diag_x_0_list = G.get_eids(diag_x_0, directed=True, error=False)
    diag_not_x_0_list = G.get_eids(diag_not_x_0, directed=True, error=False)
    #clear edges that do not exist
    diag_x_0_list = list(filter((-1).__ne__, diag_x_0_list))
    diag_not_x_0_list = list(filter((-1).__ne__, diag_not_x_0_list))

    #set edge capacities
    #TODO: clamp Q with the correct values v_i, v_j
    G.es[diag_x_0_list]["capacity"] = ...
    G.es[diag_not_x_0_list]["capacity"] = ...

    c = Q[i,i] * v_i + Q[j,j] * v_j + Q[i,j] * v_i * v_j
    return G, c
    
def compare_graphs(G_adapted, G_truth):
    # Create a set of edges for both graphs
    edges_G1 = [(e.source, e.target) for e in G_adapted.es]
    edges_G2 = [(e.source, e.target) for e in G_truth.es]
    
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

def test_adapt_graph(n, num=10000):
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
    return None, None
"""
for i in range(40):
    G_adapted, newG_truth = test_adapt_graph(4)
    if not G_adapted is None:
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

"""
Q = qubo(np.array([[-0.0347891 , -0.40028512, -0.68155756,  0.96315303],
       [ 0.        , -0.6173675 ,  0.87557738, -0.47181135],
       [ 0.        ,  0.        , -0.03672531, -0.99585558],
       [ 0.        ,  0.        ,  0.        , -0.69890477]]))
print("Input:\n", Q)
P, const = Q.to_posiform()
G_orig = to_flow_graph(P)
print("Posiform:\n", P)
#i = 0
j = 3
Q.m[:, j] = 0
print("changedQ:\n", Q)
print("changedPosiform:\n", Q.to_posiform())

print("Original graph:")
for e in G_orig.es:
    print(e.source, e.target, e["capacity"])
print("+"*20)
print("Truth graph: ")
G_truth = to_flow_graph(Q.to_posiform()[0])
for e in G_truth.es:
    print(e.source, e.target, e["capacity"])

print("My graph:")
G_adapted, c_adapted = clamp_graph(Q, G_orig, (0,j))
for e in G_adapted.es:
    print(e.source, e.target, e["capacity"])
print("DIff:", compare_graphs(G_adapted, G_truth))"""

def _remove_variable_from_posiform(P, c, changeidx, value):
    """Input: Posiform P, constant c, change indices (i, j), values of i and j
    Output: Clamped Posiform P"""
    v_i = value
    i = changeidx
    P_changed = P.copy()
    #removing row i:
    P_changed[0, i, :] = 0
    P_changed[1, i, :] = 0
    #calculate the new diagonal elements
    new_diag_elements = np.diag(P_changed[1, :, :]) - P_changed[1, :, i]
    negative_array = np.where(new_diag_elements > 0, new_diag_elements, 0)
    positive_array = np.where(new_diag_elements < 0, -new_diag_elements, 0)
    #removing column i:
    P_changed[0, :, i] = 0
    P_changed[1, :, i] = 0
    #correcting the diagonal elements
    np.fill_diagonal(P_changed[0], np.diag(P_changed[0]) + positive_array)
    np.fill_diagonal(P_changed[1], negative_array)
    P_changed[1, i, i] = np.sum(P_changed[1, i, i+1:])
    
    c = ...
    return P_changed, c

def _clamp_posiform(P, c, changeidx, values):
    i, j = changeidx
    v_i, v_j = values
    newp, newc = _remove_variable_from_posiform(P, c, i, v_i)
    newP, newC = _remove_variable_from_posiform(newp, newc, j, v_j)
    return newP, newC

def test_clamp_posiform(n, num=10000):
    for i in tqdm(range(num)):
        Q = qubo.random(n=n, distr='uniform', low=-1, high=1)
        Q_orig = Q.copy()
        P, _ = Q.to_posiform()
        change_indices = np.random.randint(0, Q.n, 2)
        i = change_indices[0]
        j = change_indices[1]
        if i > j:
            j, i = i, j
        Q.m[i,:] = 0
        Q.m[:,j] = 0
        P_changed = _clamp_posiform(P, (i,j))
        d = np.linalg.norm(Q.to_posiform()[0] - P_changed)
        if not np.isclose(d, 0):
            print("Input: \n")
            print("Q: \n", Q_orig)
            print("Posiform: \n", P)
            print("+"*10)
            print("Applying clamp:\n")
            print("Indices: \n", i, j)
            print("ChangedQ: \n", Q)
            print("+"*10)
            print("True Posiform: \n", Q.to_posiform()[0])
            print("Predicted Posiform: \n", P_changed)
            print("Diff: \n", d)
            return Q.to_posiform()[0], P_changed
    return None, None

"""
for i in range(2, 40):
    a, b = test_clamp_posiform(i) 
    if a is not None or b is not None:
        print("Help")
"""

Q = qubo(np.array([[-0.88635144, -0.4690281 ,  0.90938295,  0.72549942],
       [ 0.        , -0.21676543, -0.60432979,  0.20200042],
       [ 0.        ,  0.        ,  0.54590842, -0.93813196],
       [ 0.        ,  0.        ,  0.        , -0.36718008]]))

print("Input:\n", Q)
P, c = Q.to_posiform()
print("Original Posiform:\n", P, c)
i, j = (1, 2)
v_i, v_j = (0, 1)
assingment_str = 'x' + str(i) + '=' + str(v_i) + '; x' + str(j) + '=' + str(v_j)
PA = assignment.partial_assignment(assingment_str, n=4)
print(PA)
newQ, c = PA.apply(Q)
print("NewQ:\n", newQ)
newP, newC = newQ.to_posiform()
print("Posiform after clamping i and j: \n", newP, newC)
P_changed, c_changed = _clamp_posiform(P, c, (i, j), (v_i, v_j))
print("Predicted Posiform after clamping i and j: \n", P_changed)
P_p = np.delete(np.delete(P_changed[0], i, axis=0), i, axis=1)
P_P = np.delete(np.delete(P_p, j-1, axis=0), j-1, axis=1)
P_n = np.delete(np.delete(P_changed[1], i, axis=0), i, axis=1)
P_N = np.delete(np.delete(P_n, j-1, axis=0), j-1, axis=1)
print("P_changed;", P_P, P_N)
print("Diff P:", np.linalg.norm(P_P - newP[0]) +  np.linalg.norm(P_N - newP[1]))
#print("Diff c:", abs(c_changed - newC))
#TODO: find out how to make _clamp_posiform work with other than i=0, j=0
#ask yourself what changes when setting e.g. j=1. Look at the formulas
#should be similar to j=0 but with more change in the diagonal elements