""" Graph layout algorithms."""

import copy
from random import random

import numpy as np
from scipy.optimize import minimize
import networkx as nx


def rescale_coords(pos, scale=1, center=0):
    node_list = pos.keys()
    # dict to array
    npos = np.row_stack((pos[x] for x in node_list))
    npos = npos.astype(np.float64)
    npos = nx.rescale_layout(npos, scale=scale) + center
    # array to dict
    return dict(zip(node_list, npos))


def init_pos(G):
    return dict(zip(G.nodes.keys(), np.random.rand(len(G.nodes), 2)))


# ## ------------------------------------------
# ## ------ push / pull layout ----------------
# ## ------------------------------------------


def determine_domain(pos):
    """Determine the size of the domain.
    
    The domain size is the maximum value of a coordinate, 
    resp. the negative domain size the minimum value of a coordinate.
    Default domain size is 1. This result in a canvas of [-1, 1] in x and y direction. 
    
    :param pos: dict node to coordinates.
    :return: domain size
    """
    dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
    if dom_size == 0:
        dom_size = 1
    return dom_size


def _weight(weight, edge):
    """Determine effective weight value."""
    if isinstance(weight, float) or isinstance(weight, int):
        return float(weight)
    if isinstance(weight, str) and weight in edge:
        return edge[weight]
    return 1.0


def _mass(mass, g_node):
    """Determine effective mass of node."""
    if isinstance(mass, float) or isinstance(mass, int):
        return float(mass)
    if isinstance(mass, str) and mass in g_node:
        return g_node[mass]
    return 1.0


def edge_pull(G, node, pos, weight):
    """Determine the offset introduced by edges on node.
    
    :param G: the graph.
    :param node: key of the node.
    :param pos: dictionary keys to positions.
    :param weight: edge attribute to consider as factor or None.
    
    :return: the replacement vector on node.
    """
    n_pos = pos[node].copy()
    p_off = np.zeros(2)
    num_neighbors = len(list(G.neighbors(node)))
    for kn in G.neighbors(node):
        weight = _weight(weight, G.edges[node, kn])
        v_diff = pos[kn] - pos[node]
        p_off += v_diff / ( num_neighbors * 1.0 * weight)
    n_pos += p_off
    return n_pos


def node_push(G, node, pos, mass):
    """Determine the offset vector introduced on node by other nodes.
    
    :param G: the graph
    :param node: key node
    :param pos: dictionary node keys to positions.
    :param mass: node attribute to consider as factor or None.

    :return: the offset vector.
    """
    domain = determine_domain(pos)
    epsilon = domain / 100.0
    n_pos = pos[node].copy()
    for kn in G.nodes():
        if kn != node: 
            m1 = _mass(mass, G[node])
            m2 = _mass(mass, G[kn])
            distance = np.linalg.norm(pos[node] - pos[kn])
            unit_vdir = (pos[node] - pos[kn]) / max(distance, epsilon)
            n_pos += unit_vdir / ((domain / 25 + distance) * 1.0 * m1 * m2)
    return n_pos


def _pushpull_layout(G, pos=None, iterations=10, mass='mass', weight='weight', diagnostics=False):
    pos_history = []
    if pos is None:
        pos_ = init_pos(G)
    else:
        pos_ = copy.deepcopy(pos)
    for _ in range(iterations):
        pos_next = {}
        for node in pos_.keys():
            pos_next[node] = node_push(G, node, pos_, mass)
        pos_ = rescale_coords(pos_next)
        #pos_ = pos_next
        for node in pos_.keys():
            pos_next[node] = edge_pull(G, node, pos_, weight)
        pos_ = rescale_coords(pos_next)
        if diagnostics:
            pos_history.append(pos_)
    return pos_, pos_history


def pushpull_layout(G, pos=None, iterations=10, mass='mass', weight='weight'):
    """Push / pull layout algorithm.
    
    :param G: the graph.
    :param pos: optional dictionary with pre-defined node positions.
    :param iterations: number of iterations to apply algorithm to graph.
    :param mass: optional node attribute providing a mass.
    :param weight: optional edge attribute providing a weight.
    :return: new node positions
    """
    new_pos, _ = _pushpull_layout(G, pos=pos, iterations=iterations, mass=mass, weight=weight)
    return new_pos


# ## ---------------------------------------------------
# ## ------ path length variance layout ----------------
# ## ---------------------------------------------------


def minimize_path_length(pos_arr, edge_idx, method, tol):
    def variance_path_length(pos_arr, edge_idx):
        """Calculate variance of lengths of edges.
        
        Note: pos_arr is a 1D array, so the position of node coordinates must be calculated accordingly.
        
        :param pos_arr: 1D array of node coordinates x_0, y_0, x_1, y_1, ...
        :param edge_idx: list of edges described by indices into pos_arr.
        :return: variance edge lengths.
        """
        return np.var(np.asarray([ np.linalg.norm(np.asarray([pos_arr[i1], pos_arr[i1+1]]) - 
                                                  np.asarray([pos_arr[i2], pos_arr[i2+1]])) 
                                  for i1, i2 in edge_idx ]))


    fargs = (edge_idx, )
    ores = minimize(variance_path_length, pos_arr, fargs, method=method, tol=tol)
    if not ores.success:
        raise Exception(str(ores.message))
    return ores.x


def _pathlength_layout(G, pos=None, method='L-BFGS-B', tol=0.005, diagnostics=False):
    """Provide a layout with harmonized path lengths.
    
    :param G: graph to work on.
    :param pos: optional dictionary with pre-defined node positions.
    :param method: optimzation method to use (see scipy.optimize.minimize).
    :param tol: intensity of minimization, very small numbers will give edges of equal length.
    :param diagnostics: if True a history of node positions for each iteration is provided.
    """
    pos_history = []
    if pos is None:
        pos_ = init_pos(G)
    else:
        pos_ = copy.deepcopy(pos)
        
    # create map from node key to index into 2D array
    node_list = pos.keys()
    node_map = dict(zip(node_list, range(len(node_list))))
    # Pre-compute indices of edge nodes into pos_arr.
    edge_idx = [(node_map[n1] * 2, node_map[n2] * 2) for n1, n2 in G.edges]
    pos_arr = np.row_stack((pos[x] for x in node_list))
    pos_arr = pos_arr.astype(np.float64)

    pos_arr = minimize_path_length(pos_arr, edge_idx, method=method, tol=tol)
            
    # reshape back to 2D array ...
    npos = pos_arr.reshape(-1, 2)
    if diagnostics:
        pos_history.append(npos)
    # ... and return as pos dict
    return dict(zip(node_list, npos)), pos_history



def pathlength_layout(G, pos=None, method='L-BFGS-B', tol=0.005):
    """Provide a layout with harmonized path lengths.
    
    :param G: graph to work on.
    :param pos: optional dictionary with pre-defined node positions.
    :param method: optimzation method to use (see scipy.optimize.minimize).
    :param tol: intensity of minimization, very small numbers will give edges of equal length.
    :return: the node positions
    """
    new_pos, _ = _pathlength_layout(G, pos=pos, method=method, tot=tol)
    return new_pos
