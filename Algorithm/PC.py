import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators import PC
import pandas as pd

from Code.Algorithm import metrics


def calculate_pc(df, correct_graph):
    c = PC(df)
    model = c.estimate()

    shd = metrics.structural_hamming_distance(correct_graph.edges, model.edges)

    # Draw the estimated graph
    fig = plt.figure()
    ax_gen = fig.add_subplot(1, 2, 1)
    pos = nx.shell_layout(model)
    nx.draw_networkx_labels(model, pos)
    nx.draw_networkx_nodes(model, pos, node_size=800, alpha=0.5)
    nx.draw_networkx_edges(model, pos, alpha=0.5)

    # Draw the correct graph
    ax_corr = fig.add_subplot(1, 2, 2)
    plt.sca(ax_corr)
    plt.title(f"correct graph \n shd score (0=good 1=bad) = {shd}")

    pos = nx.shell_layout(correct_graph)
    nx.draw_networkx_labels(correct_graph, pos)
    nx.draw_networkx_nodes(correct_graph, pos, node_size=800, alpha=0.5)
    nx.draw_networkx_edges(correct_graph, pos, alpha=0.5)
    plt.show()
