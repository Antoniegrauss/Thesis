# Full Simulation

This directory contains a full simulation for causal discovery with HBCD.
Running `run.py` starts a simulation with a robot in 1 of x rooms.
The amount of rooms is dependent on which environment is being used.
The script will try to find all the causal links, showing the graph while it finds links.
The results will be written to the `Result` directory.

Afterwards the `plot_scores.py` and `plot_results.py` can be run to analyse the results.