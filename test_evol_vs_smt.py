import os
import numpy as np
import pandas as pd
from MS3_PO_MT import MS3_PO_MT
import networkx as nx

# Helper to parse a steinX.txt file
def parse_stein_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n_vertices, n_edges = map(int, lines[0].split())
    edge_lines = lines[1:1+n_edges]
    edges = []
    for line in edge_lines:
        u, v, cost = map(int, line.split())
        edges.append((u, v, cost))
    n_terminals = int(lines[1+n_edges])
    # Read terminals from all remaining lines
    terminal_lines = lines[2+n_edges:]
    terminals = []
    for line in terminal_lines:
        terminals.extend(map(int, line.split()))
    return n_vertices, n_edges, edges, terminals

# Build adjacency matrix from edge list
def build_graph(n_vertices, edges):
    G = np.full((n_vertices, n_vertices), np.inf)
    for u, v, cost in edges:
        G[u-1, v-1] = cost
        G[v-1, u-1] = cost
    return G

# Compute SMT using NetworkX's steiner_tree (approximate)
SMT = {
    'b': {1 : 82.0, 2: 83.0, 3: 138.0, 4: 59.0, 5: 61.0, 6: 122.0, 7: 111.0, 8: 104.0, 9: 220.0, 10: 86.0,
           11: 88.0, 12: 174.0, 13: 165.0, 14: 235.0, 15: 318.0, 16: 127.0, 17: 131.0, 18: 218.0},    
    'c': {1: 85.0, 2: 144.0, 3: 754.0, 4: 1079.0, 5: 1579.0, 6: 55.0, 7: 102.0, 8: 509.0, 9: 707.0, 10: 1093.0,
              11: 32.0, 12: 46.0, 13: 258.0, 14: 323.0, 15: 556.0, 16: 11.0, 17: 18.0, 18: 113.0, 19: 146.0, 20: 267.0},
    'd': {1: 106.0, 2: 220.0, 3: 1565.0, 4: 1935.0, 5: 3250.0, 6: 67.0, 7: 103.0, 8: 1072.0, 9: 1448.0, 10: 2110.0,
              11: 29.0, 12: 42.0, 13: 500.0, 14: 667.0, 15: 1116.0, 16: 13.0, 17: 23.0, 18: 223.0, 19: 310.0, 20: 537.0},
    'e': {1: 111.0, 2: 214.0, 3: 4013.0, 4: 5101.0, 5: 8128.0, 6: 73.0, 7: 145.0, 8: 2640.0, 9: 3604.0, 10: 5600.0,
              11: 34.0, 12: 67.0, 13: 1280.0, 14: 1732.0, 15: 2784.0, 16: 15.0, 17: 25.0, 18: 564.0, 19: 758.0, 20: 1342.0}
}

if __name__ == '__main__':
    results = []
    test_folder = os.path.join(os.path.dirname(__file__), 'tests')
    # Loop through X in 'b' to 'e' and Y in 1 to 20 (to 18 for 'b')
    for X in ['b','e']:
        max_Y = 18 if X == 'b' else 20
        for Y in range(1, max_Y + 1):
            fname = f'stein{X}{Y}.txt'
            fpath = os.path.join(test_folder, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                n_vertices, n_edges, edges, terminals = parse_stein_file(fpath)
                G = build_graph(n_vertices, edges)
                # Run MS3_PO_MT_EVOL
                blob = MS3_PO_MT(G, set([t-1 for t in terminals]),
                                 M=33,
                                 K=1000,
                                 alpha=0.15,
                                 mu=1,
                                 delta=0.1,
                                 S=3,
                                 évol=True,
                                 modeRenfo='vieillesse',
                                 modeProba='weighted')
                mask = np.isfinite(blob)
                blob_weight = np.sum(G[mask])/2
                # Compute SMT (approximate)
                smt_weight = SMT[X][Y] if X in SMT and Y in SMT[X] else np.inf
                error = round(max(blob_weight - smt_weight, 0)/smt_weight *100,2)
                results.append({
                    'file': fname,
                    'n_vertices': n_vertices,
                    'n_edges': n_edges,
                    'evol_weight': blob_weight,
                    'smt_weight': smt_weight,
                    'error': error
                })
                print(f"{fname}: Evol={blob_weight:.2f}, SMT={smt_weight:.2f}, Error={error:.3f}%")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
        # Save results to CSV
        pd.DataFrame(results).to_csv('evol_vs_smt_results_on_b_instance.csv', index=False)
        print("Results saved to evol_vs_smt_results.csv")
