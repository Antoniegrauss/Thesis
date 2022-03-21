import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.estimators import PC
import pandas as pd


# Run PC algorithm on the df
# save the result
# plot the resulting graph
def run_pc_plot_save(df, save_name='light_switches_pc_edges.npy', save=True, plot=True):
    c = PC(df)
    pdag = c.estimate()
    if save:
        np.save(save_name, pdag.edges)
    if plot:
        plt.figure()
        plt.title("Peter-Clarke Discovery")
        nx.draw_circular(pdag, with_labels=True, arrowsize=30, node_size=800, alpha=0.5)
        #plt.show()


# fit FCI
# G = fci(data_integer_value)
# from causallearn.utils.GraphUtils import GraphUtils
# pdy = GraphUtils.to_pydot(G)
# pdy.write_png('simple_test.png')
def main():
    five_light_switches = pd.read_csv('five_switches.csv')
    df_switches = five_light_switches
    df_switches = df_switches.drop(["action"], axis=1)
    run_pc_plot_save(df_switches, save_name='light_switches_pc_edges.npy')


if __name__ == '__main__':
    main()
