################################################
'''util functions for grid network generation'''
################################################

### created by Ricky Huang ###
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from gurobipy import *


###################
###################
'''digraph class'''
###################
###################


class digraph():
    
    def __init__(self,W,C=None,G=None,G_C=None,name='test',
                 isgrid=False,mcol=0,nrow=0,max_flow=0):
        self.W = W #weighted adjacency matrix
        self.C = C
        self.cap = C[C>0]
        self.graph = G
        self.Cgraph = G_C
        self.isgrid = isgrid # is a grid network
        if isinstance(G,nx.classes.digraph.DiGraph):
            self.n = G.number_of_nodes()
            self.m = G.number_of_edges()
        
        if isgrid:
            assert(mcol>0 and nrow>0)
            self.mcol = mcol
            self.nrow = nrow
            self.getMaps()
            self.getNxtEdges()
            self.getAllSP()
        
        else:
            self.n = len(W[0])
            self.m = 0
        self.name = name
        self.max_flow = max_flow #max_flow w.r.t. edge capacity
        self.hasMaps = False
        self.hasNxtEdges = False
        self.hasSPaths = False
        
    def getMaps(self):
        self.coordMap = coord2ind(self.W)
        self.indMap = ind2coord(self.W)
        self.hasMaps = True
        if self.m == 0:
            self.m = len(self.indMap) #edge number
        self.M = getMtrx(self.W,self.indMap) #delta-diff conservation mtrx
         
    def getNxtEdges(self):
        assert(self.hasMaps)
        self.nxtEdges = getNxtEdges(self)
        self.hasNxtEdges = True
        
    def getAllSP(self):
        assert(self.hasMaps)
        SPaths, distMatrix = allShortestPath(self)
        self.SPaths = SPaths
        self.distMatrix = distMatrix
        self.hasSPaths = True
        
    def getPathWeight(self,f):
        # f is a flow vector, f_i > 0 -> ith edge is chosen
        indx = np.where(f > 0)[0]
        out = 0
        for ind in indx:
            out += self.W[self.indMap[ind]]
        return out


def initialize(mcol,nrow,d,name=None,plot=False,
               weight_bd=None,capacity_bd=None):
    
    if weight_bd is None: a = 5; b = 10
    else: 
        assert(len(weight_bd)==2)
        a,b = weight_bd
    if capacity_bd is None: ca = int(d*0.8); cd = int(1.2*d)
    else: 
        assert(len(capacity_bd)==2)
        ca,cb = capacity_bd
    
    grid_digraph = generate_grid_network(mcol, nrow, d=d, name=name,
                   a=a, b=b, capacity_min=int(d*0.8), capacity_max=int(1.2*d))
    if plot:
        visualize_grid_network(grid_digraph.graph, mcol, nrow, 
                               show_attribute='weight') 
        visualize_grid_network(grid_digraph.graph, mcol, nrow, 
                               show_attribute='capacity') 
    grid_digraph.getMaps()
    grid_digraph.getNxtEdges()
    grid_digraph.getAllSP()
    print(f'|V|={grid_digraph.n},|E|={grid_digraph.m},edgeWeight_total={np.sum(grid_digraph.W)}')
    return grid_digraph
    

#############################
#############################
'''grid network generation'''
#############################
#############################   
    

def generate_grid_network(mcol, nrow, d=10, name='',
                          a=1, b=10, capacity_min=1, capacity_max=20):
    """
    Generates a directed grid network with the specified dimensions and edge properties.
    
    Parameters:
    m (int): Number of columns (m+1) in the grid.
    n (int): Number of rows (n+1) in the grid.
    a (int): Minimum weight for edges.
    b (int): Maximum weight for edges.
    capacity_min (int): Minimum capacity for edges.
    capacity_max (int): Maximum capacity for edges.
    
    Returns:
    G (digraph class): A directed grid network in digraph.
    """
    # Initialize directed graph
    G = nx.DiGraph()
    node_list = np.arange(mcol*nrow)
    # Add nodes and edges with weights and capacities
    assign_weights(G,mcol,nrow,d,a,b,capacity_min,capacity_max)
    
    W_matrix = np.array(nx.adjacency_matrix(G, weight='weight',nodelist=node_list).todense())
    C_matrix = np.array(nx.adjacency_matrix(G, weight='capacity',nodelist=node_list).todense())
    
    graph = digraph(W=W_matrix,C=C_matrix,G=G,name=name,isgrid=True,mcol=mcol,nrow=nrow)
    
    return graph

def assign_weights(G, mcol, nrow, d=10,
                   a=1, b=10, capacity_min=5, capacity_max=15):
    # assign random weight/capacity s.t. max flow satisfies d=flow demand
    maxFlow = 0
    while maxFlow < d:
        G.clear_edges()
        G.clear()
        G.add_node(0)
        for y in range(nrow):
            for x in range(mcol):
                # Add current node
                ind = y*mcol+x

                if x < mcol-1: # Add directed edge to the right
                    weight = random.randint(a, b)
                    capacity = random.randint(capacity_min, capacity_max)
                    G.add_edge(ind, ind+1, weight=weight, capacity=capacity)

                if x > 0: # Add directed edge to the left
                    weight = random.randint(a, b)
                    capacity = random.randint(capacity_min, capacity_max)
                    G.add_edge(ind, ind-1, weight=weight, capacity=capacity)
                
                if y < nrow-1: # Add directed edge to up
                    weight = random.randint(a, b)
                    capacity = random.randint(capacity_min, capacity_max)
                    G.add_edge(ind, ind+mcol, weight=weight, capacity=capacity)

                if y > 0: # Add directed edge to down
                    weight = random.randint(a, b)
                    capacity = random.randint(capacity_min, capacity_max)
                    G.add_edge(ind, ind-mcol, weight=weight, capacity=capacity)
        
        maxFlow, _ = nx.maximum_flow(G, 0, nrow*mcol-1, capacity='capacity')
        
    return 1


def visualize_grid_network(G, mcols, nrows, show_attribute=None, color_dict=None):
    """
    Visualizes the grid network with bidirectional edges
    and optional edge attributes (weights/capacity/flow/delta).
    
    Parameters:
    G (nx.DiGraph): The directed grid network.
    show_attribute (str): Edge attribute to display
    """
    pos = {node: (node % mcols, node // mcols) for node in G.nodes()}  
    base_size = int(7*max(1,mcols*nrows/20))
    offset = 0.05*min(1,80/(mcols*nrows)) # Offset amount for bidirectional edges
    fsize = int(12*min(1, 120/(mcols*nrows)))#edge label font size
    fig, ax = plt.subplots(figsize=(
        int(base_size*mcols/nrows) if mcols > nrows else base_size,
        int(base_size*nrows/mcols) if mcols <= nrows else base_size
    ))
    #Color dictionary
    if color_dict is None:
        color_dict = dict()
        color_dict['nd_bg'] = 'black'; color_dict['nd_txt'] = 'white'
        color_dict['ar_fill'] = 'gray'; color_dict['ar_edge'] = 'gray'
        color_dict['ed_wt'] = 'gray'
    
    nx.draw_networkx_nodes(G, pos, node_size=350, node_color=color_dict['nd_bg'], ax=ax)
    
    for u, v in G.edges():
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        edge_length = max(np.sqrt(dx**2 + dy**2), 0.01)  #Avoid division by zero
        dir_x, dir_y = dx/edge_length, dy/edge_length  #Normalized direction and perpendicular vectors
        perp_x, perp_y = -dir_y, dir_x  #Perpendicular direction
        
        plt.arrow(
            pos[u][0] + offset * dy, pos[u][1] - offset * dx, dx * 0.8, dy * 0.8,
            head_width=0.05, head_length=0.05, 
            fc=color_dict['ar_fill'], ec=color_dict['ar_edge'], length_includes_head=False
            )
        if show_attribute: #draw edge label
            pos_x = (pos[u][0] + pos[v][0])/2 
            pos_y = (pos[u][1] + pos[v][1])/2
            deg = np.degrees(np.arctan2(dy, dx))
            if dy==0: 
                deg = 0
                pos_y -= 2*offset*dx
            elif dx==0:
                pos_x += 2*offset*dy  
            label_pos_forward = (pos_x,pos_y)
            ax.text(*label_pos_forward, str(G[u][v][show_attribute]),
                   rotation=deg,
                   rotation_mode='anchor',
                   fontsize=fsize, alpha=0.9,
                   color=color_dict['ed_wt'],  # Match edge color
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    if show_attribute:
        plt.title(f"{nrows}x{mcols} network with '{show_attribute}'", fontsize=14)
    else:
        plt.title(f"{nrows}x{mcols} network", fontsize=14)

    # Display the graph
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    
def visualize_grid_flow(grid_digraph, f, mcols, nrows, title=''):
    # grid_digraph: digraph object
    # Gcopy: grid nx.classes.digraph.DiGraph
    # f: (gurobi-) solved min-cost flow

    Gcopy = grid_digraph.graph.copy()
    cdmap = grid_digraph.coordMap
    pos = {node: (node % mcols, node // mcols) for node in Gcopy.nodes()} 
    base_size = int(7*max(1,mcols*nrows/20))
    offset = 0.05*min(1,80/(mcols*nrows)) # Offset amount for bidirectional edges
    fsize = int(12*min(1, 120/(mcols*nrows)))#edge label font size
    fig, ax = plt.subplots(figsize=(
        int(base_size*mcols/nrows) if mcols > nrows else base_size,
        int(base_size*nrows/mcols) if mcols <= nrows else base_size
    ))
    # Draw the nodes
    nx.draw_networkx_nodes(Gcopy, pos, node_size=400, node_color='black')

    # Draw edges with offset for bidirectional edges
    for u, v in Gcopy.edges():
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]

        if f[cdmap[(u,v)]] > 0:
            Gcopy[u][v]['flow'] = round(f[cdmap[(u,v)]],2)
            color = 'red'
        else:
            color = 'gray'
        plt.arrow(
            pos[u][0] + offset * dy, pos[u][1] - offset * dx, dx * 0.8, dy * 0.8,
            head_width=0.05, head_length=0.05, 
            fc=color, ec=color, length_includes_head=False
        )
        if color == "red":
            pos_x = (pos[u][0] + pos[v][0])/2 
            pos_y = (pos[u][1] + pos[v][1])/2
            deg = np.degrees(np.arctan2(dy, dx))
            if dy==0: 
                deg = 0
                pos_y -= 2*offset*dx
            elif dx==0:
                pos_x += 2*offset*dy  
            label_pos_forward = (pos_x,pos_y)
            ax.text(*label_pos_forward, str(Gcopy[u][v]['flow']),
                   rotation=deg,
                   rotation_mode='anchor',
                   fontsize=fsize, alpha=0.9,
                   color=color,  # Match edge color
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    plt.title(f"{nrows}x{mcols} network with '{title}'", fontsize=14)
    # Display the graph
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_grid_delta(grid_digraph, mcols, nrows, W_delta, HL_delta, title=''):
    # grid_digraph: digraph object
    # W_delta: weighted adjacency matrix applied with perturbation delta
    # HL_delta: highlighted edges with perturbed weight (increased, decreased, mixed)
    
    assert(len(HL_delta)==3)
    Gcopy = grid_digraph.graph.copy()
    pos = {node: (node % mcols, node // mcols) for node in Gcopy.nodes()} 
    for i, j in Gcopy.edges():
        Gcopy[i][j]['delta'] = round(W_delta[i, j],3)
    
    base_size = int(7*max(1,mcols*nrows/20))
    offset = 0.05*min(1,80/(mcols*nrows)) # Offset amount for bidirectional edges
    fsize = int(12*min(1, 120/(mcols*nrows)))#edge label font size
    fig, ax = plt.subplots(figsize=(
        int(base_size*mcols/nrows) if mcols > nrows else base_size,
        int(base_size*nrows/mcols) if mcols <= nrows else base_size
    ))
    # Draw the nodes
    nx.draw_networkx_nodes(Gcopy, pos, node_size=400, node_color='black')
        
    # Draw edges with offset for bidirectional edges
    colors = ['red','green','orange','gray']
    for u, v in Gcopy.edges():
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
            
        uv_color = 'gray'
        for i in range(3):
            if (u,v) in HL_delta[i]: 
                uv_color = colors[i]
        plt.arrow(
            pos[u][0] + offset * dy, pos[u][1] - offset * dx, dx * 0.8, dy * 0.8,
            head_width=0.05, head_length=0.05, 
            fc=uv_color, ec=uv_color, length_includes_head=False
        )
        pos_x = (pos[u][0] + pos[v][0])/2 
        pos_y = (pos[u][1] + pos[v][1])/2
        deg = np.degrees(np.arctan2(dy, dx))
        if dy==0: 
            deg = 0
            pos_y -= 2*offset*dx
        elif dx==0:
            pos_x += 2*offset*dy  
        label_pos_forward = (pos_x,pos_y)
        ax.text(*label_pos_forward, str(Gcopy[u][v]['delta']),
               rotation=deg,
               rotation_mode='anchor',
               fontsize=fsize, alpha=0.9,
               color=uv_color,  # Match edge color
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title(f"{nrows}x{mcols} network with '{title}'", fontsize=14)
    # Display the graph
    plt.axis("off")
    plt.tight_layout()
    plt.show()
        
    
    
########################
########################
'''for graph indexing'''
########################
########################


def coord2ind(A):
    # A: adjacency matrix
    # maps edge (i,j) to an index
    # returns a dict{edge:index}
    n = A.shape[0]
    ct = -1
    mapp = {}
    for i in range(n):
        for j in range(n):
            if A[i,j] > 0:
                ct += 1
                mapp[(i,j)]=ct
    return mapp

def ind2coord(A):
    # A: adjacency matrix
    # maps index to edge (i,j)
    # returns a dict{index:coord}
    n = A.shape[0]
    ct = -1
    mapp = {}
    for i in range(n):
        for j in range(n):
            if A[i,j] > 0:
                ct += 1
                mapp[ct]=(i,j)
    return mapp


def getMtrx(W,indMap):
    # W: adjacency matrix
    # indMap: dict{ind:edge(i,j)}
    # returns the delta-conservation matrix
    n = W.shape[0]
    m = len(indMap)
    res = np.zeros((n,2*m))
    for i in range(n): #for every vtx
        for key,value in indMap.items():
            if value[0] == i: #edge (i,j)
                    res[i,key] = 1
            elif value[1] == i:
                    res[i,key+m] = -1 #edge (k,i)
    return res


def getRvecs(W,indMap):
    # W: adjacency matrix
    # indMap: dict{ind:edge(i,j)}
    # returns the source req vector
    # assuming 0=source, n-1=sink
    n = W.shape[0]
    m = len(indMap)
    rs = np.zeros(m); rt = np.zeros(m)
    for key,value in indMap.items():
            if value[0] == 0: #edge (0,j)
                rs[key] = 1
            if value[1] == n-1: #edge (k,n-1)
                rt[key] = 1
    return rs, rt


def getA_v2(W,indMap):
    # W: adjacency matrix
    # indMap: dict{ind:edge(i,j)}
    # returns the incidence matrix
    n = W.shape[0]
    m = len(indMap)
    res = np.zeros((n,m))
    for key, value in indMap.items():
        res[value[0],key] = -1  #edge key starts with value[0]
        res[value[1],key] = 1  #edge key ends with value[1]
        
    return res

def getNxtEdges(G):
    # returns a dict{edge:all its receiving edges}
    out = defaultdict(list)
    for edge in G.coordMap:
        row = G.W[edge[1],:]
        for i in range(G.n):
            if row[i] > 0:
                out[edge].append((edge[1],i))
            
    return out

#########################
'''Floyd–Warshall algo'''
#########################
#finds shortest paths of all pairs of vtx
#Complexity O(|V|^3)

def allShortestPath(G):
    #G: digraph object
    paths = defaultdict(list)
    distMatrix = np.ones((G.n,G.n))*max(np.sum(G.W),1e6)
    for i in range(G.n):
        distMatrix[i,i] = 0
        paths[(i,i)] = []
    for edge in G.coordMap:
        distMatrix[edge] = G.W[edge]
        paths[edge] = [edge]
    for k in range(G.n):
        for l in range(G.n):
            for j in range(G.n):
                if distMatrix[(l,j)] > distMatrix[(l,k)] + distMatrix[(k,j)]:
                    distMatrix[(l,j)] = distMatrix[(l,k)] + distMatrix[(k,j)]
                    paths[(l,j)] = paths[(l,k)] + paths[(k,j)]
    return paths, distMatrix

#######################
'''Flow-testing algo'''
#######################

def UnrobustModel(W,d,cap,Layer=None,plotG=False):
    #no robust at all, baseline
    f_out,opm_val = optimizeNormal(W,d,cap,name=None)
    return f_out, opm_val

def optimizeNormal(W,d=10,k=1.0,name=None,
                    pout= False):
    # W: adjacency matrix
    # d: flow amount requirement
    # k: egde flow capacity

    indMap = ind2coord(W)
    m = len(indMap)
    A = getA_v2(W,indMap)
    rs,rt = getRvecs(W,indMap)
    cost = W[W > 0]
    if name is None:
        name = "test"
    model = Model(name)
    
    if not pout:
        model.Params.LogToConsole = 0
    
    f = model.addMVar(m, lb=0, name="flow")
    f_out = []

    rhs = np.zeros(W.shape[0])
    rhs[0] = -d; rhs[-1] = d
    model.update()
    model.addConstr(A@f == rhs, name="flow conservation")
    model.addConstr(rs@f == d, name="source flow req")
    model.addConstr(rt@f == d, name="sink flow req")
    model.addConstr(f <= k, name="capacity req")
    model.setObjective(cost@f, GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print(W)
        print(A)
        print(rhs)
        print(rs)
        print(rt)
    
    opm_val = model.getObjective().getValue()
    
    if pout:
        print("\n---Output:---\n")
        model.printAttr('x')
        print(f"min cost is {opm_val}.\n")
        
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    for name, val in zip(names, values):
#         print(f"{name} = {val}")
        if "flow" in name:
            f_out.append(val)
    return np.array(f_out), opm_val

















