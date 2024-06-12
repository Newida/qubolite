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

def adapt_graph(P, G_orig, change_idx):
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
    
    #convert edges to edge_ids
    diag_x_0_list = G.get_eids(diag_x_0, directed=True, error=False)
    diag_not_x_0_list = G.get_eids(diag_not_x_0, directed=True, error=False)
    #clear edges that do not exist
    diag_x_0_list = list(filter((-1).__ne__, diag_x_0_list))
    diag_not_x_0_list = list(filter((-1).__ne__, diag_not_x_0_list))

    #set edge capacities
    #TODO: clamp Q with the correct values v_i, v_j
    G.es[diag_x_0_list]["capacity"] = np.diag(P[0])
    G.es[diag_not_x_0_list]["capacity"] = np.diag(P[1])

    c = 0 #use partial assignment to calculate the constant
    #or do it yourself
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

def _clamp_posiform(P, changeidx):
    i, j = changeidx
    P_changed = P.copy()
    #removing i:
    P_changed[0, i, :] = 0
    P_changed[1, i, :] = 0
    #calculate the new diagonal elements
    diag_elements = np.diag(P[1, :, :]) - P[1, :, j]
    negative_array = np.where(diag_elements > 0, diag_elements, 0)
    positive_array = np.where(diag_elements < 0, -diag_elements, 0)
    #removing j:
    P_changed[0, :, j] = 0
    P_changed[1, :, j] = 0

    #correcting the diagonal elements
    np.fill_diagonal(P_changed[0], np.diag(P_changed[0]) + positive_array)
    np.fill_diagonal(P_changed[1], negative_array)
    return P_changed

def test_clamp_posiform(n, num=10000):
    for i in tqdm(range(num)):
        Q_orig = qubo.random(n=n, distr='uniform', low=-1, high=1)
        P, _ = Q_orig.to_posiform()
        Q = Q_orig.copy()
        change_indices = np.random.randint(0, Q.n, 2)
        i = change_indices[0]
        j = change_indices[1]
        if i > j:
            j, i = i, j
        Q.m[:,j] = 0
        Q.m[i,:] = 0
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

#test_clamp_posiform(4, 3)
Q = qubo(np.array([[-0.08011814,  0.07762449, -0.57877048, -0.69547865],
       [ 0.        , -0.31275536,  0.09832495, -0.99704806],
       [ 0.        ,  0.        ,  0.02660838,  0.00245066],
       [ 0.        ,  0.        ,  0.        ,  0.63774563]]))
P = Q.to_posiform()[0]
i, j = (1, 2)
Q.m[:, j] = 0
Q.m[i, :] = 0
print("True Posiform: \n", Q.to_posiform()[0])
P_changed = _clamp_posiform(P, (i, j))
print("Diff:", np.linalg.norm(P_changed - Q.to_posiform()[0]))
"""
Q = qubo(np.array([[-0.08011814,  0.07762449, -0.57877048, -0.69547865],
       [ 0.        , -0.31275536,  0.09832495, -0.99704806],
       [ 0.        ,  0.        ,  0.02660838,  0.00245066],
       [ 0.        ,  0.        ,  0.        ,  0.63774563]]))
print("Input:\n", Q)
P, const = Q.to_posiform()
G_orig = to_flow_graph(P)
print("Posiform:\n", P)
j = 2
Q.m[:, j] = 0
print("changedQ:\n", Q)
print("changedPosiform:\n", Q.to_posiform())
P_predicted = P.copy()
#calculate the new diagonal elements
diag_elements = np.diag(P_predicted[1, :, :]) - P_predicted[1, :, j]
# Create the first array with positive elements and 0s
negative_array = np.where(diag_elements > 0, diag_elements, 0)
# Create the second array with negative elements and 0s
positive_array = np.where(diag_elements < 0, -diag_elements, 0)
P_predicted[0, :, j] = 0
P_predicted[1, :, j] = 0
np.fill_diagonal(P_predicted[0], np.diag(P_predicted[0]) + positive_array)
np.fill_diagonal(P_predicted[1], negative_array)
print("Predicted Posiform:\n", P_predicted)
print("Diff:", np.linalg.norm(P_predicted - Q.to_posiform()[0]))
"""
"""
Q = qubo(np.array([[-0.0347891 , -0.40028512, -0.68155756,  0.96315303],
       [ 0.        , -0.6173675 ,  0.87557738, -0.47181135],
       [ 0.        ,  0.        , -0.03672531, -0.99585558],
       [ 0.        ,  0.        ,  0.        , -0.69890477]]))
print("Input:\n", Q)
P, const = Q.to_posiform()
print("Posiform:\n", P)
i = 3
#j = 3
Q.m[i, :] = 0
print("changedQ:\n", Q)
print("changedPosiform:\n", Q.to_posiform())
P_predicted = P.copy()
P_predicted[0, i, :] = 0
P_predicted[1, i, :] = 0
print("predicted P:", P_predicted)
print("Diff:", np.linalg.norm(P_predicted - Q.to_posiform()[0]))
"""