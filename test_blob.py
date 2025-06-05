import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import sys
import os
import functools # Import functools

# Ensure Fonctions directory is in the path if MS3_PO_MT is run directly
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Optional: Adjust if needed

# Assuming Blobv3.py and Fonctions are accessible
from MS3_PO import MS3_PO
from MS3_PO_MT import MS3_PO_MT# Import Blob and potentially poids if needed by Blobv3
from Fonctions.Tools import nx2np # Import the conversion utility

# --- Global variables for callback counter and plot ---
fig, ax = plt.subplots(figsize=(8, 8))

# --- Matplotlib Callback (Accepts step_index) ---
# Add step_index argument, remove blob_state (it's the first arg passed by partial)
def matplotlib_callback(blob_state, step_index, base_pos, terminal_nodes, all_nodes):
    """Visualizes the blob state using matplotlib at a specific step."""
    global fig, ax # Keep these globals

    # Optional: Add a check to draw less frequently if desired, based on step_index
    if step_index % 20 != 0:
         return # Only draw every 20 steps

    plt.clf() # Clear the current figure

    # 1. Extract edges and their conductivities from blob_state
    blob_edges_with_weights = [] # Store tuples (u, v, weight)
    num_nodes_state = blob_state.shape[0]
    threshold = 1e-6 # Threshold to consider an edge 'present'

    # Ensure index mapping matches if nodes were relabeled
    # Assuming nodes 0 to num_nodes-1 correspond directly to matrix indices
    if num_nodes_state != len(all_nodes):
         # This warning check is now valid again
         print(f"Warning: Blob state size {num_nodes_state} != node list size {len(all_nodes)}")
         # Adjust logic if necessary, here we assume direct mapping

    for i in range(num_nodes_state):
        for j in range(i + 1, num_nodes_state):
            conductivity = blob_state[i, j]
            if np.isfinite(conductivity) and conductivity > threshold:
                 # Check if nodes i and j exist in our base_pos mapping
                 if i in base_pos and j in base_pos:
                     blob_edges_with_weights.append((i, j, conductivity))

    # 2. Draw the graph components
    # Draw all nodes
    nx.draw_networkx_nodes(all_nodes, base_pos, node_size=20, node_color='lightblue', alpha=0.8)

    # Draw terminal nodes distinctly
    nx.draw_networkx_nodes(terminal_nodes, base_pos, node_size=50, node_color='red')

    # Draw blob edges and labels
    if blob_edges_with_weights:
        # Create a temporary graph for drawing edges
        blob_graph_viz = nx.Graph()
        blob_graph_viz.add_nodes_from(all_nodes) # Add all nodes for consistent drawing
        # Extract just edges for drawing lines
        blob_edges = [(u, v) for u, v, w in blob_edges_with_weights]
        blob_graph_viz.add_edges_from(blob_edges)
        nx.draw_networkx_edges(blob_graph_viz, base_pos, edgelist=blob_edges, edge_color='green', width=2.0)

        # Create edge labels dictionary for conductivity display
        edge_labels = {(u, v): f"{w:.2f}" for u, v, w in blob_edges_with_weights}
        nx.draw_networkx_edge_labels(blob_graph_viz, base_pos, edge_labels=edge_labels, font_size=7, font_color='darkgreen')


    # Use step_index in the title
    plt.title(f"Blob State - Step {step_index}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.pause(0.5) # Pause for 0.5 seconds to update the plot


# --- Main Test Logic ---
if __name__ == "__main__":

    print("--- Testing Blob Function with Grid Graph and Matplotlib Viz ---")

    # 1. Create Grid Graph
    grid_n = 12 # Number of rows - Check if this matches the expected 25 nodes? If 5x5 -> 25 nodes.
    grid_p = 12 # Number of columns
    num_total_nodes = grid_n * grid_p
    print(f"Creating a {grid_n}x{grid_p} grid graph ({num_total_nodes} nodes).")

    g_nx = nx.grid_2d_graph(grid_n, grid_p)

    # Create positions dictionary (mapping node ID to coordinates)
    # Node (r, c) -> position [c, grid_n - 1 - r] for standard plot orientation
    pos_tuples = {(r, c): np.array([c, grid_n - 1 - r]) for r in range(grid_n) for c in range(grid_p)}

    # Relabel nodes from (r,c) tuples to integers 0 to N-1
    node_mapping = {(r, c): r * grid_p + c for r in range(grid_n) for c in range(grid_p)}
    g_nx_relabeled = nx.relabel_nodes(g_nx, node_mapping, copy=True)

    # Define context locally
    pos = {node_mapping[old_node]: pos_tuples[old_node] for old_node in pos_tuples}
    all_node_indices = list(g_nx_relabeled.nodes())

    # Add weights (distances) to edges for the NumPy conversion
    for u, v in g_nx_relabeled.edges():
        p1 = pos[u]
        p2 = pos[v]
        distance = math.sqrt(((p1 - p2)**2).sum())
        g_nx_relabeled.edges[u, v]['weight'] = distance

    # Convert to NumPy adjacency matrix
    sample_graph = nx2np(g_nx_relabeled) # Use the relabeled graph
    if sample_graph.shape[0] != num_total_nodes:
        print(f"Warning: nx2np output shape {sample_graph.shape} mismatch with expected {num_total_nodes}")

    # 2. Define Terminals (Randomly)
    num_terminals = 5
    if num_terminals > num_total_nodes:
        num_terminals = num_total_nodes
    # Define terminals locally
    terminals = set(random.sample(all_node_indices, num_terminals))
    print(f"Terminals (randomly selected): {terminals}")

    # 3. Set parameters
    M_test = 10
    K_test = 400 # Number of steps for visualization
    alpha_test = 0.7
    mu_test = 1.0
    delta_test = 0.2
    epsilon_test = 1e-5

    print(f"Running Blob with M={M_test}, K={K_test}...")


    # 4. Prepare Matplotlib and Callback using functools.partial
    plt.ion()
    # Update partial call: bind only context args
    # The resulting viz_callback will expect (blob_state, step_index)
    viz_callback = functools.partial(matplotlib_callback,
                                     base_pos=pos,
                                     terminal_nodes=terminals,
                                     all_nodes=all_node_indices)

    # 5. Call Blob function
    final_blob = MS3_PO_MT( # This now returns the MST of the best blob
        Graphe=sample_graph.copy(),
        Terminaux=terminals,
        M=M_test,
        K=K_test,
        alpha=alpha_test,
        mu=mu_test,
        delta=delta_test,
        epsilon=epsilon_test,
        display_result=True, # Use our callback for display
        step_callback=viz_callback, # Pass the partial object
        modeRenfo='vieillesse'
    )


    # 6. Final Plot and Results
    plt.ioff() # Turn off interactive mode
    print("\n--- Blob Function Test Result ---")
    if final_blob is not None:
        print(f"Returned MST Matrix Shape: {final_blob.shape}") # Changed print statement

        # Visualize the final MST
        plt.clf() # Clear the figure used for steps

        # Extract MST edges from the final_blob matrix
        mst_edges = []
        mst_edge_labels = {}
        num_nodes_final = final_blob.shape[0]
        for i in range(num_nodes_final):
            for j in range(i + 1, num_nodes_final):
                weight = final_blob[i, j]
                # Check if it's a valid edge in the MST matrix (not inf)
                if np.isfinite(weight):
                    # Use local pos here
                    if i in pos and j in pos:
                        mst_edges.append((i, j))
                        mst_edge_labels[(i, j)] = f"{weight:.2f}" # Store weight for label

        # Draw all nodes (use local all_node_indices, pos)
        nx.draw_networkx_nodes(all_node_indices, pos, node_size=20, node_color='lightblue', alpha=0.8)
        # Draw terminal nodes (use local terminals, pos)
        nx.draw_networkx_nodes(terminals, pos, node_size=50, node_color='red')
        # Draw MST edges
        if mst_edges:
            mst_graph_viz = nx.Graph()
            mst_graph_viz.add_nodes_from(all_node_indices)
            mst_graph_viz.add_edges_from(mst_edges)
            nx.draw_networkx_edges(mst_graph_viz, pos, edgelist=mst_edges, edge_color='blue', width=2.5)
            # Draw MST edge labels (optional)
            nx.draw_networkx_edge_labels(mst_graph_viz, pos, edge_labels=mst_edge_labels, font_size=7, font_color='darkblue')

        plt.title(f"Final MST Result (After {K_test + K_test//3} Blob steps)")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show() # Keep the final plot window open

    else:
        print("Blob function returned None.")
        plt.show() # Show empty plot if needed

    print("\n--- Test Complete ---")
