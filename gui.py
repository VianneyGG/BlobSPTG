import tkinter as tk
from tkinter import ttk # Corrected: Style is part of ttk
import math
from itertools import combinations
import random
import networkx as nx # Import networkx
import numpy as np # Import numpy
import concurrent.futures # For threading
import queue # For thread communication
import os # To get CPU count
from tkinter import messagebox # Import messagebox for showing errors

# --- Import Blob algorithm and helpers ---
# Ensure Blobv3.py and the Fonctions directory are in the Python path
try:
    from MS3_PO import MS3_PO # Updated import
    from Fonctions.Outils import nx2np
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import Blob algorithm components: {e}\nMake sure Blobv3.py and Fonctions directory are accessible.")

# Reference values for scaling
DEFAULT_GRID_SIZE = 10
DEFAULT_CELL_SIZE = 40
DEFAULT_NODE_RADIUS_FACTOR = 8 / DEFAULT_CELL_SIZE # Node radius as fraction of cell size
DEFAULT_PLACEHOLDER_RADIUS_FACTOR = 3 / DEFAULT_CELL_SIZE # Placeholder radius as fraction of cell size

# Target initial canvas size
INITIAL_CANVAS_WIDTH = DEFAULT_GRID_SIZE * DEFAULT_CELL_SIZE
INITIAL_CANVAS_HEIGHT = DEFAULT_GRID_SIZE * DEFAULT_CELL_SIZE

MIN_GRID_SIZE = 2
MAX_GRID_SIZE = 30
MAX_RADIUS = 5 # Define the maximum radius limit

class GraphGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Graph Grid Interaction")

        # Get screen dimensions
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        # Calculate 80% of screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Set initial window size and position (centered)
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        self.master.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

        # Colors
        self.window_bg = "#2c3e50" # Dark blue for main window and canvas background
        self.menu_bg = "#34495e" # Slightly lighter dark blue for menu
        self.grid_placeholder_color = "#34495e" # Lighter blue for grid lines and placeholders
        self.menu_fg = "white"    # White text for contrast
        self.slider_trough_color = "#566573" # Keep this or adjust if needed
        self.entry_bg = "#566573" # Background for entry fields
        self.separator_color = "#7f8c8d" # Color for separators
        self.button_active_bg = "#4a6175" # Slightly lighter blue for active button

        # Set master background color
        self.master.config(bg=self.window_bg)

        # Grid dimensions and scaling
        global INITIAL_CANVAS_WIDTH, INITIAL_CANVAS_HEIGHT
        menu_approx_width = 180 # Estimate for menu width + padding
        padding_approx = 40 # Includes padding around frames
        INITIAL_CANVAS_WIDTH = window_width - menu_approx_width - padding_approx
        INITIAL_CANVAS_HEIGHT = window_height - padding_approx * 1.5 # Reduce vertical padding impact slightly

        self.grid_size = DEFAULT_GRID_SIZE
        self.calculate_scaling_factors()

        initial_intersections = (DEFAULT_GRID_SIZE + 1) ** 2
        default_random_nodes = int(0.8 * initial_intersections)

        # Tkinter variables
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        self.radius_var = tk.DoubleVar(value=1.0)
        self.random_nodes_var = tk.IntVar(value=default_random_nodes)

        # Data structures
        self.nodes = {}
        self.terminals = set()
        self.placeholders = {}
        self.offset_x = 0 # Offset for centering grid horizontally
        self.offset_y = 0 # Offset for centering grid vertically
        self.mst_queue = queue.Queue() # Queue for MST result
        self.blob_step_queue = queue.Queue() # Queue for intermediate Blob steps
        self.blob_running = False # Flag to indicate if Blob is running
        self.current_index_to_coord = {} # To map indices back in step checker
        num_workers = os.cpu_count() or 1 # Default to 1 if cpu_count returns None or 0
        print(f"Using {num_workers} workers for thread pool.")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) # Thread pool

        self.grid_size_var.trace_add("write", self.update_grid_size)
        self.setup_ui()

    def setup_ui(self):
        # --- Style Configuration ---
        style = ttk.Style() # Corrected: Instantiate from ttk.Style
        style.theme_use('clam') # Use a theme that allows more customization

        # Configure Button style
        style.configure('Menu.TButton',
                        background=self.menu_bg,
                        foreground=self.menu_fg,
                        borderwidth=1,
                        focusthickness=3,
                        focuscolor='none') # Remove focus highlight if desired
        style.map('Menu.TButton',
                  background=[('active', self.button_active_bg), ('pressed', self.window_bg)],
                  foreground=[('pressed', self.menu_fg), ('active', self.menu_fg)])

        # Configure Entry style
        style.configure('Menu.TEntry',
                        fieldbackground=self.entry_bg,
                        foreground=self.menu_fg, # Text color
                        insertcolor=self.menu_fg, # Cursor color
                        borderwidth=1) # Adjust border as needed

        # Configure Separator style
        style.configure('Menu.TSeparator', background=self.separator_color)

        # Main frame - Set background to window background
        main_frame = tk.Frame(self.master, bg=self.window_bg)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10) # Add padding via grid
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Menu Frame (Left) - Set background to lighter blue
        menu_frame = tk.Frame(main_frame, width=150, bg=self.menu_bg) # Use lighter blue for menu
        menu_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        menu_frame.pack_propagate(False)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)

        # Grid Frame (Right) - Set background to window background
        grid_frame = tk.Frame(main_frame, bg=self.window_bg)
        grid_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0,10), pady=10) # Add padding via grid
        main_frame.columnconfigure(1, weight=1)

        # --- Menu Widgets ---
        # Apply the custom style to buttons
        clear_button = ttk.Button(menu_frame, text="Clear Grid", command=self.clear_grid, style='Menu.TButton')
        clear_button.pack(pady=5, fill=tk.X, padx=5)

        # Apply the custom style to separators
        ttk.Separator(menu_frame, orient='horizontal', style='Menu.TSeparator').pack(pady=10, fill=tk.X, padx=5)

        # Grid Size Label, Entry, and Slider
        grid_size_label = ttk.Label(menu_frame, text="Grid Size:", background=self.menu_bg, foreground=self.menu_fg)
        grid_size_label.pack(pady=(5, 0))

        # Apply the custom style to entries
        grid_size_entry = ttk.Entry(
            menu_frame,
            textvariable=self.grid_size_var,
            width=5,
            style='Menu.TEntry'
        )
        grid_size_entry.pack(pady=(0, 5))

        self.grid_size_slider = tk.Scale(
            menu_frame,
            from_=MIN_GRID_SIZE,
            to=MAX_GRID_SIZE,
            orient=tk.HORIZONTAL,
            variable=self.grid_size_var,
            resolution=1,
            length=130,
            showvalue=0,
            bg=self.menu_bg, # Match menu background
            fg=self.menu_fg,
            highlightthickness=0,
            troughcolor=self.slider_trough_color
        )
        self.grid_size_slider.pack(pady=(0, 10))

        # Apply the custom style to separators
        ttk.Separator(menu_frame, orient='horizontal', style='Menu.TSeparator').pack(pady=10, fill=tk.X, padx=5)

        # Radius Label and Slider
        radius_label = ttk.Label(menu_frame, text="Connection Radius:", background=self.menu_bg, foreground=self.menu_fg)
        radius_label.pack(pady=(5, 0))

        self.radius_slider = tk.Scale(
            menu_frame,
            from_=1,
            to=MAX_RADIUS, # Set maximum value to MAX_RADIUS
            orient=tk.HORIZONTAL,
            variable=self.radius_var,
            resolution=1, # Keep resolution at 1 for integer steps
            length=130,
            command=self.update_edges,
            bg=self.menu_bg, # Match menu background
            fg=self.menu_fg,
            highlightthickness=0,
            troughcolor=self.slider_trough_color
        )
        self.radius_slider.pack(pady=(0, 5))

        # Apply the custom style to separators
        ttk.Separator(menu_frame, orient='horizontal', style='Menu.TSeparator').pack(pady=10, fill=tk.X, padx=5)

        # Random Nodes Generation
        random_nodes_label = ttk.Label(menu_frame, text="Random Nodes:", background=self.menu_bg, foreground=self.menu_fg)
        random_nodes_label.pack(pady=(5, 0))

        # Apply the custom style to entries
        random_nodes_entry = ttk.Entry(
            menu_frame,
            textvariable=self.random_nodes_var,
            width=5,
            style='Menu.TEntry'
        )
        random_nodes_entry.pack(pady=(0, 5))

        # Apply the custom style to buttons
        generate_button = ttk.Button(
            menu_frame,
            text="Generate Random",
            command=self.generate_random_nodes,
            style='Menu.TButton'
        )
        generate_button.pack(pady=5, fill=tk.X, padx=5)

        # Add Fill Grid button
        fill_button = ttk.Button(
            menu_frame,
            text="Fill Grid",
            command=self.fill_grid,
            style='Menu.TButton'
        )
        fill_button.pack(pady=5, fill=tk.X, padx=5)

        # Apply the custom style to separators
        ttk.Separator(menu_frame, orient='horizontal', style='Menu.TSeparator').pack(pady=10, fill=tk.X, padx=5) # Added separator

        # Add Run Blob button
        blob_button = ttk.Button(
            menu_frame,
            text="Run Blob",
            command=self.run_blob_algorithm, # Link to the new method
            style='Menu.TButton'
        )
        blob_button.pack(pady=5, fill=tk.X, padx=5)

        # --- Grid Canvas ---
        self.canvas = tk.Canvas(grid_frame, width=INITIAL_CANVAS_WIDTH, height=INITIAL_CANVAS_HEIGHT, bg=self.window_bg, borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.draw_grid()
        self.update_edges()

        # Bind click event
        self.canvas.bind("<Button-1>", self.handle_click)

    def calculate_scaling_factors(self):
        """Calculates cell_size, node_radius, and placeholder_radius based on grid_size and canvas dimensions."""
        # Calculate required size factor including grid cells and radius margins
        size_factor = self.grid_size + 2 * DEFAULT_NODE_RADIUS_FACTOR

        # Determine cell_size based on the limiting dimension
        if size_factor > 0: # Avoid division by zero
             cell_size_w = INITIAL_CANVAS_WIDTH / size_factor
             cell_size_h = INITIAL_CANVAS_HEIGHT / size_factor
             self.cell_size = min(cell_size_w, cell_size_h)
        else: # Should not happen with grid_size >= MIN_GRID_SIZE
             self.cell_size = min(INITIAL_CANVAS_WIDTH / self.grid_size, INITIAL_CANVAS_HEIGHT / self.grid_size)

        # Ensure cell_size is positive
        self.cell_size = max(1, self.cell_size) # Minimum cell size of 1 pixel

        # Recalculate radii based on the determined cell size
        self.node_radius = DEFAULT_NODE_RADIUS_FACTOR * self.cell_size
        self.placeholder_radius = DEFAULT_PLACEHOLDER_RADIUS_FACTOR * self.cell_size
        print(f"Calculated: Cell Size={self.cell_size:.2f}, Node Radius={self.node_radius:.2f}")

    def update_grid_size(self, *args):
        try:
            new_size = self.grid_size_var.get()
            if not (MIN_GRID_SIZE <= new_size <= MAX_GRID_SIZE):
                print(f"Invalid grid size: {new_size}. Must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}.")
                self.grid_size_var.set(self.grid_size)
                return

            if new_size != self.grid_size:
                print(f"Updating grid size to: {new_size}x{new_size}")
                self.grid_size = new_size

                self.calculate_scaling_factors()

                # Radius slider 'to' value is now fixed at MAX_RADIUS, no need to update it here.
                # Just ensure the current value doesn't exceed the max.
                if self.radius_var.get() > MAX_RADIUS:
                    self.radius_var.set(MAX_RADIUS)

                new_intersections = (self.grid_size + 1) ** 2
                new_default_random_nodes = int(0.8 * new_intersections)
                self.random_nodes_var.set(new_default_random_nodes)
                print(f"Updated default random nodes to: {new_default_random_nodes}")

                self.clear_grid()

        except tk.TclError:
            print("Invalid input for grid size.")

    def draw_grid(self):
        self.canvas.delete("all")
        self.placeholders.clear()

        # Calculate total pixel dimensions of the grid itself (intersection to intersection)
        grid_pixel_width = self.grid_size * self.cell_size
        grid_pixel_height = self.grid_size * self.cell_size

        # Calculate offsets to center the grid area, ensuring node radius margin is respected
        self.offset_x = (INITIAL_CANVAS_WIDTH - grid_pixel_width) / 2
        self.offset_y = (INITIAL_CANVAS_HEIGHT - grid_pixel_height) / 2

        # Ensure offsets leave space for node radius (at least node_radius from canvas edge)
        self.offset_x = max(self.node_radius, self.offset_x)
        self.offset_y = max(self.node_radius, self.offset_y)

        # Check if calculated offsets push the grid beyond the canvas limits due to radius constraint
        if self.offset_x + grid_pixel_width + self.node_radius > INITIAL_CANVAS_WIDTH:
             self.offset_x = INITIAL_CANVAS_WIDTH - grid_pixel_width - self.node_radius
             self.offset_x = max(self.node_radius, self.offset_x)

        if self.offset_y + grid_pixel_height + self.node_radius > INITIAL_CANVAS_HEIGHT:
             self.offset_y = INITIAL_CANVAS_HEIGHT - grid_pixel_height - self.node_radius
             self.offset_y = max(self.node_radius, self.offset_y)

        # Draw placeholders centered using the final offsets
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size + 1):
                x = c * self.cell_size + self.offset_x
                y = r * self.cell_size + self.offset_y
                ph_id = self.canvas.create_oval(
                    x - self.placeholder_radius, y - self.placeholder_radius,
                    x + self.placeholder_radius, y + self.placeholder_radius,
                    fill=self.grid_placeholder_color, outline=self.grid_placeholder_color, tags="placeholder"
                )
                self.placeholders[(r, c)] = ph_id

        # Draw grid lines centered using the final offsets
        grid_start_x = self.offset_x
        grid_end_x = grid_start_x + grid_pixel_width
        grid_start_y = self.offset_y
        grid_end_y = grid_start_y + grid_pixel_height

        for i in range(self.grid_size + 1):
            x = i * self.cell_size + self.offset_x
            self.canvas.create_line(x, grid_start_y, x, grid_end_y, fill=self.grid_placeholder_color)
            y = i * self.cell_size + self.offset_y
            self.canvas.create_line(grid_start_x, y, grid_end_x, y, fill=self.grid_placeholder_color)

    def handle_click(self, event):
        grid_x = event.x - self.offset_x
        grid_y = event.y - self.offset_y

        grid_width = self.grid_size * self.cell_size
        grid_height = self.grid_size * self.cell_size
        tolerance = self.cell_size / 2.5 # Define tolerance early

        # Adjust boundary check to include tolerance around the grid edges
        if not (-tolerance <= grid_x <= grid_width + tolerance and
                -tolerance <= grid_y <= grid_height + tolerance):
            return # Exit if click is too far outside the grid + tolerance area

        # Use floating point division for potentially more accurate rounding near edges
        col_f = grid_x / self.cell_size
        row_f = grid_y / self.cell_size
        col = round(col_f)
        row = round(row_f)

        # Ensure calculated row/col are within valid grid indices before proceeding
        if not (0 <= row <= self.grid_size and 0 <= col <= self.grid_size):
            return # Click rounded to an invalid intersection index

        target_x = col * self.cell_size
        target_y = row * self.cell_size

        # Check if click is within tolerance of the rounded intersection point
        if abs(grid_x - target_x) < tolerance and abs(grid_y - target_y) < tolerance:
            coord = (row, col)
            if coord not in self.nodes:
                self.add_node(coord)
            elif self.nodes[coord]['state'] == 'node':
                self.add_terminal(coord)
            elif self.nodes[coord]['state'] == 'terminal':
                self.remove_node(coord)

    def add_node(self, coord):
        row, col = coord
        # Calculate center based on grid coords and add offset for drawing
        center_x = col * self.cell_size + self.offset_x
        center_y = row * self.cell_size + self.offset_y
        # Use node radius divided by 1.20 for black nodes
        draw_radius = self.node_radius / 1.20
        x1 = center_x - draw_radius
        y1 = center_y - draw_radius
        x2 = center_x + draw_radius
        y2 = center_y + draw_radius

        if coord in self.placeholders:
            self.canvas.itemconfig(self.placeholders[coord], state='hidden')

        node_id = self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black', tags="node")
        self.nodes[coord] = {'id': node_id, 'state': 'node'}
        print(f"Node added at intersection: {coord}")
        self.update_edges()

    def add_terminal(self, coord):
        if coord in self.nodes and self.nodes[coord]['state'] == 'node':
            node_info = self.nodes[coord]
            terminal_color = "#c0392b" # Darker red color
            self.canvas.itemconfig(node_info['id'], fill=terminal_color, outline=terminal_color)
            node_info['state'] = 'terminal'
            self.terminals.add(coord)
            print(f"Node at {coord} marked as terminal.")
            print(f"Current Terminals: {self.terminals}")

    def remove_node(self, coord):
        if coord in self.nodes:
            node_info = self.nodes[coord]
            self.canvas.delete(node_info['id'])
            del self.nodes[coord]

            if coord in self.terminals:
                self.terminals.remove(coord)

            if coord in self.placeholders:
                self.canvas.itemconfig(self.placeholders[coord], state='normal')

            print(f"Node removed from intersection: {coord}")
            print(f"Current Terminals: {self.terminals}")
            self.update_edges()
            self.canvas.delete("blob_edge") # Clear blob result if graph changes

    def build_current_graph(self):
        """Builds a NetworkX graph from current nodes and edges based on radius."""
        g = nx.Graph()
        node_list = list(self.nodes.keys())
        coord_to_index = {coord: i for i, coord in enumerate(node_list)}
        index_to_coord = {i: coord for i, coord in enumerate(node_list)}

        # Add nodes with integer indices
        for i in range(len(node_list)):
            g.add_node(i)

        # Add edges based on radius, using integer indices and distance as weight
        radius = self.radius_var.get()
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                coord1 = node_list[i]
                coord2 = node_list[j]
                r1, c1 = coord1
                r2, c2 = coord2
                distance = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

                if 0 < distance <= radius:
                    g.add_edge(coord_to_index[coord1], coord_to_index[coord2], weight=distance)

        return g, index_to_coord

    def run_blob_algorithm(self):
        """Builds the graph, runs Blob algorithm in a thread, and handles result."""
        if self.blob_running:
            messagebox.showwarning("Busy", "Blob algorithm is already running.")
            return

        if len(self.terminals) < 2:
            messagebox.showerror("Error", "Blob algorithm requires at least 2 terminals (red nodes).")
            print("Blob algorithm requires at least 2 terminals.")
            return

        if not self.nodes:
             messagebox.showerror("Error", "No nodes on the grid.")
             print("Cannot run Blob on an empty grid.")
             return

        print("Preparing graph for Blob algorithm...")
        self.canvas.delete("blob_edge") # Clear previous results

        try:
            graph, index_to_coord = self.build_current_graph()
            if graph.number_of_nodes() == 0:
                 messagebox.showerror("Error", "Graph construction failed (no nodes).")
                 return

            np_graph = nx2np(graph) # Convert to NumPy adjacency matrix
            terminal_coords = list(self.terminals)
            coord_to_index = {coord: i for i, coord in index_to_coord.items()}
            terminal_indices = [coord_to_index[t] for t in terminal_coords if t in coord_to_index]

            if len(terminal_indices) < 2:
                 messagebox.showerror("Error", "Could not map terminals to graph nodes.")
                 return

            self.current_index_to_coord = index_to_coord.copy()

            def gui_step_callback(intermediate_blob_state):
                if self.blob_running:
                    self.blob_step_queue.put(intermediate_blob_state.copy())

            print(f"Running Blob with {np_graph.shape[0]} nodes and terminals: {terminal_indices}")
            self.blob_running = True
            while not self.blob_step_queue.empty():
                try: self.blob_step_queue.get_nowait()
                except queue.Empty: break

            future = self.executor.submit(
                Blob,
                np_graph,
                set(terminal_indices),
                M=1,
                K=1000,
                display_result=False,
                step_callback=gui_step_callback
            )

            future.add_done_callback(
                lambda f: self._handle_blob_result(f)
            )
            self.master.after(50, self._check_blob_steps)
            print("Blob algorithm submitted. Monitoring steps...")

        except NameError as e:
             messagebox.showerror("Error", f"Blob function or helper not found: {e}. Check imports.")
             self.blob_running = False
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during Blob preparation: {e}")
            print(f"Error preparing Blob: {e}")
            self.blob_running = False

    def _check_blob_steps(self):
        """Periodically checks the queue for intermediate blob results and updates canvas."""
        if not self.blob_running:
            print("Blob step checking stopped.")
            return

        try:
            processed_step = False
            while True:
                intermediate_blob_np = self.blob_step_queue.get_nowait()
                processed_step = True

                result_edges = []
                num_nodes = intermediate_blob_np.shape[0]
                index_to_coord = self.current_index_to_coord
                if num_nodes != len(index_to_coord):
                    print(f"Warning: Intermediate matrix size ({num_nodes}) doesn't match node mapping ({len(index_to_coord)}). Skipping step.")
                    continue

                threshold = 1e-6

                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        if np.isfinite(intermediate_blob_np[i, j]) and intermediate_blob_np[i, j] > threshold:
                            coord1 = index_to_coord.get(i)
                            coord2 = index_to_coord.get(j)
                            if coord1 and coord2:
                                result_edges.append((coord1, coord2))

                if processed_step:
                    self.display_blob_result(result_edges)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing blob step: {e}")

        if self.blob_running:
            self.master.after(50, self._check_blob_steps)

    def _handle_blob_result(self, future):
        """Callback function to process the FINAL Blob result from the thread."""
        was_running = self.blob_running
        self.blob_running = False

        while not self.blob_step_queue.empty():
            try:
                self.blob_step_queue.get_nowait()
            except queue.Empty:
                break
        print("Cleared blob step queue.")

        if not was_running:
             print("Blob result received, but process was already stopped or finished.")
             return

        try:
            result_blob_np = future.result()
            print("Blob algorithm finished. Displaying final result.")

            result_edges = []
            num_nodes = result_blob_np.shape[0]
            index_to_coord = self.current_index_to_coord
            if num_nodes != len(index_to_coord):
                 print(f"Warning: Final matrix size ({num_nodes}) doesn't match node mapping ({len(index_to_coord)}).")

            threshold = 1e-6

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if np.isfinite(result_blob_np[i, j]) and result_blob_np[i, j] > threshold:
                        coord1 = index_to_coord.get(i)
                        coord2 = index_to_coord.get(j)
                        if coord1 and coord2:
                            result_edges.append((coord1, coord2))

            print(f"Final Blob result: Found {len(result_edges)} edges.")
            self.display_blob_result(result_edges)

        except Exception as e:
            messagebox.showerror("Blob Error", f"Error executing or processing final Blob result: {e}")
            print(f"Error in Blob execution/callback: {e}")
        finally:
            self.current_index_to_coord = {}

    def display_blob_result(self, edges):
        """Draws the edges resulting from the Blob algorithm."""
        self.canvas.delete("blob_edge")
        blob_edge_color = "#2ecc71"
        blob_edge_width = 3.0

        if not edges:
            return

        drawn_edges = set()
        for coord1, coord2 in edges:
            edge_tuple = tuple(sorted((coord1, coord2)))
            if edge_tuple in drawn_edges:
                continue

            if coord1 in self.nodes and coord2 in self.nodes:
                r1, c1 = coord1
                r2, c2 = coord2
                x1 = c1 * self.cell_size + self.offset_x
                y1 = r1 * self.cell_size + self.offset_y
                x2 = c2 * self.cell_size + self.offset_x
                y2 = r2 * self.cell_size + self.offset_y
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=blob_edge_color,
                    width=blob_edge_width,
                    tags="blob_edge"
                )
                drawn_edges.add(edge_tuple)

        self.canvas.tag_raise("node")
        self.canvas.tag_lower("blob_edge", "node")

    def update_edges(self, event=None):
        self.canvas.delete("edge")
        self.canvas.delete("blob_edge")
        radius = self.radius_var.get()
        node_coords = list(self.nodes.keys())
        edges_created = False
        edge_color = "#1f618d"

        for coord1, coord2 in combinations(node_coords, 2):
            r1, c1 = coord1
            r2, c2 = coord2
            distance = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

            if 0 < distance <= radius:
                x1 = c1 * self.cell_size + self.offset_x
                y1 = r1 * self.cell_size + self.offset_y
                x2 = c2 * self.cell_size + self.offset_x
                y2 = r2 * self.cell_size + self.offset_y
                self.canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=2.5, tags="edge")
                edges_created = True

        if edges_created and self.nodes:
            self.canvas.tag_lower("edge", "node")

    def clear_grid(self):
        if self.blob_running:
            print("Clearing grid: Stopping active Blob process.")
            self.blob_running = False

        while not self.blob_step_queue.empty():
            try: self.blob_step_queue.get_nowait()
            except queue.Empty: break
        while not self.mst_queue.empty():
            try: self.mst_queue.get_nowait()
            except queue.Empty: break

        self.nodes.clear()
        self.terminals.clear()
        self.canvas.delete("blob_edge")
        self.draw_grid()
        self.update_edges()
        self.current_index_to_coord = {}
        print("Grid cleared.")

    def _select_nodes_from_chunk(self, coord_chunk, num_to_select):
        """Worker function to select a random sample of nodes from a coordinate chunk."""
        if num_to_select <= 0:
            return []
        if num_to_select >= len(coord_chunk):
            return coord_chunk
        return random.sample(coord_chunk, num_to_select)

    def _calculate_mst_radius(self, node_coords):
        """Calculates the optimal radius based on MST (runs in a separate thread)."""
        if len(node_coords) < 2:
            return 1.0

        temp_g = nx.Graph()
        temp_g.add_nodes_from(node_coords)
        max_mst_edge_weight = 0

        for coord1, coord2 in combinations(node_coords, 2):
            r1, c1 = coord1
            r2, c2 = coord2
            distance = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            if distance > 0:
                temp_g.add_edge(coord1, coord2, weight=distance)

        try:
            if not nx.is_connected(temp_g):
                print("Graph is not connected, cannot compute MST reliably for radius adjustment.")
                return min(1.0, float(MAX_RADIUS))

            mst = nx.minimum_spanning_tree(temp_g, weight='weight')
            for u, v, edge_data in mst.edges(data=True):
                max_mst_edge_weight = max(max_mst_edge_weight, edge_data['weight'])

            if max_mst_edge_weight > 0:
                new_radius = math.ceil(max_mst_edge_weight)
                clamped_radius = max(1.0, min(new_radius, float(MAX_RADIUS)))
                return clamped_radius
            else:
                return min(1.0, float(MAX_RADIUS))

        except nx.NetworkXError as e:
            print(f"Error computing MST: {e}. Returning default radius 1.")
            return min(1.0, float(MAX_RADIUS))
        except Exception as e:
            print(f"Unexpected error during MST radius calculation: {e}")
            return min(1.0, float(MAX_RADIUS))

    def _check_mst_result(self):
        """Checks the queue for the MST result and updates GUI if available."""
        try:
            result_radius = self.mst_queue.get_nowait()
            print(f"MST calculation finished. Suggested radius: {result_radius}")
            self.radius_var.set(result_radius)
            self.update_edges()
            print("Radius and edges updated based on MST calculation.")
        except queue.Empty:
            self.master.after(100, self._check_mst_result)
        except Exception as e:
            print(f"Error processing MST result: {e}")
            self.radius_var.set(1.0)
            self.update_edges()

    def generate_random_nodes(self):
        self.clear_grid()

        try:
            num_nodes_target = self.random_nodes_var.get()
        except tk.TclError:
            print("Invalid number entered for random nodes.")
            return

        max_possible_nodes = (self.grid_size + 1) ** 2
        if not (0 < num_nodes_target <= max_possible_nodes):
            print(f"Number of random nodes must be between 1 and {max_possible_nodes}.")
            self.random_nodes_var.set(min(max(1, num_nodes_target), max_possible_nodes))
            num_nodes_target = self.random_nodes_var.get()

        all_coords = [(r, c) for r in range(self.grid_size + 1) for c in range(self.grid_size + 1)]

        select_to_include = num_nodes_target <= max_possible_nodes // 2
        if select_to_include:
            num_to_select_in_parallel = num_nodes_target
            print(f"Selecting {num_to_select_in_parallel} nodes to include.")
        else:
            num_to_exclude = max_possible_nodes - num_nodes_target
            num_to_select_in_parallel = num_to_exclude
            print(f"Selecting {num_to_select_in_parallel} nodes to exclude (faster).")

        random.shuffle(all_coords)

        num_workers = self.executor._max_workers
        chunk_size = len(all_coords) // num_workers
        remainder_coords = len(all_coords) % num_workers

        items_per_worker = num_to_select_in_parallel // num_workers
        remainder_items = num_to_select_in_parallel % num_workers

        futures = []
        start_index = 0
        print(f"Distributing selection task across {num_workers} workers...")
        for i in range(num_workers):
            end_index = start_index + chunk_size + (1 if i < remainder_coords else 0)
            chunk = all_coords[start_index:end_index]
            num_items_for_worker = items_per_worker + (1 if i < remainder_items else 0)

            if chunk and num_items_for_worker > 0:
                futures.append(self.executor.submit(self._select_nodes_from_chunk, chunk, num_items_for_worker))
            start_index = end_index

        parallel_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result_chunk = future.result()
                parallel_results.extend(result_chunk)
            except Exception as e:
                print(f"Error retrieving result from worker thread: {e}")

        if select_to_include:
            final_selected_coords = parallel_results
            if len(final_selected_coords) > num_nodes_target:
                 final_selected_coords = final_selected_coords[:num_nodes_target]
        else:
            excluded_coords_set = set(parallel_results)
            while len(excluded_coords_set) > num_to_exclude:
                 excluded_coords_set.pop()
            final_selected_coords = [coord for coord in all_coords if coord not in excluded_coords_set]

        print(f"Parallel selection complete. Adding {len(final_selected_coords)} nodes to the grid...")

        for coord in final_selected_coords:
            self.add_node(coord)
        print(f"Finished adding {len(self.nodes)} nodes.")

        node_coords = list(self.nodes.keys())
        if len(node_coords) >= 2:
            print("Submitting MST calculation to background thread...")
            mst_future = self.executor.submit(self._calculate_mst_radius, node_coords)

            def _mst_callback(f):
                try:
                    result = f.result()
                    self.mst_queue.put(result)
                except Exception as e:
                    print(f"Error retrieving MST result from future: {e}")
                    self.mst_queue.put(1.0)

            mst_future.add_done_callback(_mst_callback)
            self.master.after(100, self._check_mst_result)

        elif len(node_coords) <= 1:
            self.radius_var.set(1.0)
            self.update_edges()
            print("Not enough nodes for MST calculation. Set radius to 1.")

        print("Random node generation process initiated. MST calculation running in background (if applicable).")

    def fill_grid(self):
        """Fills every intersection point on the grid with a node."""
        self.clear_grid()
        print(f"Filling grid ({self.grid_size+1}x{self.grid_size+1})...")
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size + 1):
                self.add_node((r, c))
        print("Grid filled.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphGUI(root)
    def on_closing():
        print("Shutting down thread pool...")
        try:
             app.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
             app.executor.shutdown(wait=False)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
