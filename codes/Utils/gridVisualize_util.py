###################################################
'''util functions for grid network visualization'''
###################################################

### created by Ricky Huang ###
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

############################################
'''Generate simple grid network version.1'''
############################################


def grid_network_v1(mcols, nrows, d=10, name='',
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
    node_list = np.arange(mcols*nrows)
    # Add nodes and edges with weights and capacities
    assign_weights(G,mcols,nrows,d,a,b,capacity_min,capacity_max)
    
    W_matrix = np.array(nx.adjacency_matrix(G, weight='weight',nodelist=node_list).todense())
    C_matrix = np.array(nx.adjacency_matrix(G, weight='capacity',nodelist=node_list).todense())
    
    return G, W_matrix, C_matrix


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


####################################
##################################
'''Visualization util functions'''
##################################
###################################


def visualize_grid_network(G, mcols, nrows, ax,
                           pos, offset, fsize,
                           color_dict, show_attribute=None):
    """
    Visualizes the grid network with bidirectional edges
    and optional edge attributes (weights/capacity/flow/delta).
    
    Parameters:
    G (nx.DiGraph): The directed grid network.
    show_attribute (str): Edge attribute to display
    """

    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=color_dict['nd_bg'], ax=ax)
    
    for u, v in G.edges():
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        edge_length = max(np.sqrt(dx**2 + dy**2), 0.01)  #Avoid division by zero
        dir_x, dir_y = dx/edge_length, dy/edge_length  #Normalized direction and perpendicular vectors
        perp_x, perp_y = -dir_y, dir_x  #Perpendicular direction
        
        plt.arrow(
            pos[u][0] + offset * dy, pos[u][1] - offset * dx, dx * 0.9, dy * 0.9,
            head_width=0.05, head_length=0.05, 
            fc=color_dict['ar_fill'], ec=color_dict['ar_edge'], 
            length_includes_head=False
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
                   color=color_dict['ed_wt'], 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
      
    return 1


#######################################
'''Visualize grid with one car/rider'''
#######################################


class GridwithAgent:
    '''Store and draw car/rider position on a grid
       with format ('node', position_data) or 
                   ('edge', position_data)'''
    
    def __init__(self, G, mcols, nrows, color_dict=None):
        # G is a (nx.DiGraph) object
        self.G = G
        self.mcols = mcols
        self.nrows = nrows
        self.pos = {node: (node % mcols, node // mcols) for node in G.nodes()}
        self.base_sz = int(7*max(1,mcols*nrows/20)) #for plot figure size
        self.offset = 0.035*min(1,80/(mcols*nrows)) #for edge pos offset
        self.font_sz = int(12*min(1, 120/(mcols*nrows))) #for edge label font sz
        self.car_position = None; self.rdr_position = None
        
        if color_dict is None:
            #default colors
            self.color_dict = dict()
            self.color_dict['nd_bg'] = 'black'; self.color_dict['nd_txt'] = 'white'
            self.color_dict['ar_fill'] = 'gray'; self.color_dict['ar_edge'] = 'gray'
            self.color_dict['ed_wt'] = 'gray'
            self.color_dict['car'] = 'white'; self.color_dict['car_ed'] = 'red'
            self.color_dict['rdr'] = 'gold'; self.color_dict['rdr_ed'] = 'blue'
        else:
            self.color_dict = color_dict
        
    def place_on_node(self, node, kind='car'):
        '''Place car/rider exactly on a node'''
        #Note: node is also a coord (x,y)
        if kind == 'car':
            self.car_position = ('node', node)
        elif kind == 'rdr':
            self.rdr_position = ('node', node)
        
    def place_on_edge(self, u, v, frac=0.5, kind='car'):
        '''Place car along an edge (fraction=0 at u, 1 at v)'''
        assert(0 <= frac <= 1.2)
        if kind == 'car':
            self.car_position = ('edge', (u, v, frac))
        elif kind == 'rdr':
            self.rdr_position = ('edge', (u, v, frac))
        
    def get_coord(self, kind='car'):
        '''Convert car position to plot coordinates'''
        if kind == 'car':
            position = self.car_position
        elif kind == 'rdr':
            position = self.rdr_position
        
        if not position:
            return None
            
        if position[0] == 'node':
            return self.pos[position[1]]
        else: #'edge' format
            u, v, frac = position[1]
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            dx = self.pos[v][0] - self.pos[u][0]
            dy = self.pos[v][1] - self.pos[u][1]
            x_off = 0; y_off = 0
            deg = np.degrees(np.arctan2(dy, dx))
            
            if dy==0: 
                deg = 0
                y_off = -self.offset*dx
                if kind == 'rdr': y_off *= 2
            elif dx==0:
                x_off = self.offset*dy  
                if kind == 'rdr': x_off *= 2
            
            return (x1+frac*(x2-x1)+x_off, y1+frac*(y2-y1)+y_off)
    
    
    def visualize(self, show_attribute=None, 
                  car_image_path=None, rdr_image_path=None):
        '''Visualize grid with car'''
        fig, ax = plt.subplots(figsize=self.get_figure_size())

        #Draw network
        self.draw_network(ax, show_attribute)
        
        #Draw car if placed
        car_coords = self.get_coord()
        if car_coords:
            if car_image_path:
                self.draw_image(ax, car_coords, car_image_path)
            else:
                self.draw_dot(ax, car_coords)
        #Draw rider if placed
        rdr_coords = self.get_coord('rdr')
        if rdr_coords:
            if rdr_image_path:
                self.draw_image(ax, rdr_coords, rdr_image_path)
            else:
                self.draw_dot(ax, rdr_coords,'rdr')
        
        
        ax.set_title(f"{self.nrows}x{self.mcols} network" + 
                    (f" with '{show_attribute}'" if show_attribute else ""))
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    
    def get_figure_size(self):
        figsize = (
        int(self.base_sz*self.mcols/self.nrows) if self.mcols > self.nrows else self.base_sz,
        int(self.base_sz*self.nrows/self.mcols) if self.mcols <= self.nrows else self.base_sz
        )
        return figsize
    
    def draw_network(self, ax, show_attribute):
        # Draw nodes and edges (similar to your previous implementation)
        visualize_grid_network(self.G, self.mcols, self.nrows, ax,
                           self.pos, self.offset, self.font_sz,
                           self.color_dict, show_attribute)
        

    def draw_dot(self, ax, coords, kind='car'):
        ax.plot(coords[0], coords[1], 'o', 
                markersize=int(14*30/(self.mcols*self.nrows)),
                markerfacecolor=self.color_dict[kind], 
                markeredgecolor=self.color_dict[kind+'_ed'])
    
    def draw_image(self, ax, coords, image_path):
        img = plt.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, coords, frameon=False)
        ax.add_artist(ab)

    ###################
    '''For animation'''
    ###################
    
    def visualize_animation(self, path_frames, show_path=False, path_color='r', 
                            show_attribute=None, car_image_path=None, rdr_image_path=None,
                            fps=10, save_path=None):
        """
        Creates animation of car moving along path with disappearing trail.
        
        Args:
            path_frames: List from generate_animated_path()
            show_attribute: Edge attribute to display
            car_image_path/rdr_image_path: Optional image paths
            fps: Frames per second
            save_path: If provided, saves as GIF
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size())
        if show_path:
            path_line, = ax.plot([], [], path_color+':', alpha=0.85, linewidth=2)
        # Initialize plot elements
        self.draw_network(ax, show_attribute)
        
        # Pre-compute all frame data
        full_car_coords = []
        full_rdr_coords = []
        
        if show_path:
            full_path_coords = []
        for i, frame in enumerate(path_frames):
            coord = self.get_coord_from_frame(frame)
            full_car_coords.append(coord)
            if show_path:
                full_path_coords.append(coord)
                path_line, = ax.plot([],[],
                    path_color+':', alpha=0.85, linewidth=2, zorder=1
                    )
        # Draw car and rider
        car_artist = self._init_car_artist(ax, car_image_path)
        rdr_artist = self._init_rdr_artist(ax, rdr_image_path)
        
        if self.car_position:
            car_artist.zorder = 2
            ax.add_artist(car_artist)
        if self.rdr_position: 
            rdr_artist.zorder = 2
            ax.add_artist(rdr_artist)

        def update(frame):
            # Update path visualization
            if show_path:
                remaining_path = full_path_coords[frame:]
                if len(remaining_path) > 0:
                    path_line.set_data(
                        [c[0] for c in remaining_path],
                        [c[1] for c in remaining_path]
                    )
                else:
                    path_line.set_data([], [])  # Clear when reached destination
            
            # Update car
            if self.car_position:
                car_coords = full_car_coords[frame]
#                 self.car_position = car_coords
                if car_image_path:
                    car_artist.xy = car_coords
                else:
                    car_artist.set_data([car_coords[0]], [car_coords[1]])
            
            # Update rider if exists
            if self.rdr_position:
                rdr_coords = self.get_coord('rdr')
                if rdr_image_path:
                    rdr_artist.xy = rdr_coords
                else:
                    rdr_artist.set_data([rdr_coords[0]], [rdr_coords[1]])
                    
            if show_path:
                return path_line, car_artist, rdr_artist
            else: 
                return car_artist, rdr_artist
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(path_frames),
            interval=1000/fps, blit=True
        )
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=fps)
        
        return ani

    def _init_car_artist(self, ax, image_path):
        """Initialize car artist based on visualization type"""
        if image_path:
            img = plt.imread(image_path)
            imagebox = OffsetImage(img, zoom=0.1)
            return AnnotationBbox(imagebox, (0,0), frameon=False)
        else:
            return ax.plot([], [], 'o', markersize=int(14*30/(self.mcols*self.nrows)),
                          markerfacecolor=self.color_dict['car'],
                          markeredgecolor=self.color_dict['car_ed'])[0]

    def _init_rdr_artist(self, ax, image_path):
        """Initialize rider artist based on visualization type"""
        if image_path:
            img = plt.imread(image_path)
            imagebox = OffsetImage(img, zoom=0.1)
            return AnnotationBbox(imagebox, (0,0), frameon=False)
        else:
            return ax.plot([], [], 'o', markersize=int(14*30/(self.mcols*self.nrows)),
                          markerfacecolor=self.color_dict['rdr'],
                          markeredgecolor=self.color_dict['rdr_ed'])[0]

    def get_coord_from_frame(self, frame, kind='car'):
        """Helper to get coordinates from animation frame data"""
        
        if isinstance(frame, int):  # Node
            return self.pos[frame]
        elif len(frame)==3:  # Edge
            u, v, frac = frame
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            dx = self.pos[v][0] - self.pos[u][0]
            dy = self.pos[v][1] - self.pos[u][1]
            x_off = 0; y_off = 0
            deg = np.degrees(np.arctan2(dy, dx))
            
            if dy==0: 
                deg = 0
                y_off = -self.offset*dx
                if kind == 'rdr': y_off *= 2
            elif dx==0:
                x_off = self.offset*dy  
                if kind == 'rdr': x_off *= 2
            
            return (x1+frac*(x2-x1)+x_off, y1+frac*(y2-y1)+y_off)
    

#####################################
'''Location-to-path util functions'''
#####################################


def generate_animated_path(raw_path: list) -> list:
    """
    Complete pipeline from raw path to animated frames with easing effect.
    
    1. First creates unified traversal path
    2. Then generates animation frames with variable speed
    
    Args:
        raw_path: List containing nodes, edges, or edge+fractions
        
    Returns:
        List of animation frames: [(position, fraction), ...]
    """
    # Step 1: Create unified traversal path
    traversal_path = create_traversal_path(raw_path)
    
    # Step 2: Generate animated frames
    return convert_to_animation_frames(traversal_path)


def create_traversal_path(raw_path: list) -> list:
    """
    Converts various path specifications into a unified traversal sequence.
    
    Handles:
    - Node-to-node paths: [1, 2, 3] → [1, (1,2,1.0), 2, (2,3,1.0), 3]
    - Node-to-edge paths: [1, 2, (2,3,0.5)] → [1, (1,2,1.0), 2, (2,3,0.5)]
    - Edge-to-edge paths: [(1,2,0.3), (2,3,0.7)] → [(1,2,0.3), 2, (2,3,0.7)]
    - Edge-to-node paths: [(1,2,0.3), (2,3,1.0), 3] → [(1,2,0.3), 2, (2,3,1.0), 3]

    Args:
        raw_path: List containing either:
                  - Node IDs (integers)
                  - Edge tuples (u, v)
                  - Edge+position tuples (u, v, fraction)
                  
    Returns:
        Unified traversal path alternating between:
        - Nodes (integers)
        - Full edges (u, v, 1.0)
        - Partial edges (u, v, fraction)
    """
    if not raw_path:
        return []
    
    path = []
    prev_item = None
    
    for item in raw_path:
        if isinstance(item, int):  # Node
            if prev_item is None:
                path.append(item)
            elif isinstance(prev_item, int):
                path.append((prev_item,item,1.0))
            elif len(prev_item) >= 2:  # Full edge or Partial edge
                if prev_item[1] != item:
                    raise ValueError(f"Edge {prev_item} doesn't connect to edge {item}")
                path.append(item)
        
        elif isinstance(item, tuple):  # Edge
            if len(item) == 2:
                u, v = item
                item = (u,v,1.0)
            if prev_item is None:
                path.append(item)
            elif isinstance(prev_item, int):
                if prev_item != item[0]:
                    raise ValueError(f"Node {prev_item} doesn't connect to edge {item}")
                path.append(item)
            elif len(prev_item) == 3:  # Partial edge
                if prev_item[1] != item[0]:
                    raise ValueError(f"Edge {prev_item} doesn't connect to edge {item}")
                path.append(item)
        prev_item = item
    
    return path

def convert_to_animation_frames(traversal_path: list) -> list:
    """Converts unified path to animation frames with easing"""
    frames = []
    
    for i, item in enumerate(traversal_path):
        if isinstance(item, int):  # Node
            frames.append(item)
        elif len(item) == 3:
            u, v, frac = item
            if frac > 0.9: frac = 0.9
            if i == 0: 
                frames.append(item)
            else:
                frames.extend(generate_edge_frames(u, v, frac))
    return frames

def generate_edge_frames(u: int, v: int, frac: float) -> list:
    """Generates eased frames for an edge segment"""
    # Determine snapshot count
    if frac <= 0.3:
        n = 3
    elif frac <= 0.6:
        n = 6 
    else:
        n = 8
    
    # Create eased distribution (slow-fast-slow)
    t = np.linspace(0, 1, n)
    fractions = frac * (np.sin((t - 0.5) * np.pi) + 1) / 2
    
    return [(u, v, f) for f in fractions]


        
        