from pgmpy.estimators import BayesianEstimator, ExpectationMaximization
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models import BayesianNetwork
import pandas as pd
import numpy as np

data_int = np.load('int_data.npy')

'''
# Fit the pdag to a DBN
model = DBN(
    [
        (("room", 0), ("waypoint", 0)),
        (("type", 0), ("waypoint", 0)),
        (("room", 0), ("room", 1)),
        (("type", 0), ("type", 1)),
        (("waypoint", 0), ("waypoint", 1))
    ]
)
colnames = [("waypoint", 0), ("waypoint", 1), ("room", 0), ("room", 1), ("type", 0), ("type", 1)]
df_dbn = pd.DataFrame(data_int, columns=colnames)
model.fit(df_dbn)
'''

# Fit the pdag to a BN
model = BayesianNetwork(
    [
        ("room0", "room1"),
        ("type0", "type1"),
        ("type0", "changed"),
        ("type1", "changed")
    ]
)
colnames = ["waypoint0", "waypoint1", "room0", "room1", "type0", "type1", "changed"]
df_dbn = pd.DataFrame(data_int, columns=colnames)
df_dbn = df_dbn.drop(["waypoint0", "waypoint1"], axis=1)
#df_dbn.to_csv('data_int_small.csv')
model.fit(data=df_dbn)
print(df_dbn)

# Print the cpd's
for cpd in model.cpds:
    print(cpd.variables)
    values = cpd.values
    print(values.shape)
    print(np.array(values.astype(np.float)))
    print(cpd)


# Simulate data with BN
#data = pd.DataFrame(model.simulate(n_samples=1000))
#data.to_csv('BN_simulated_data.csv')