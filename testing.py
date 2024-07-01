# %%
from NNReplication import replicators

# Usage example:
# %%
model = replicators.SimpleModel()
# %%
# before regen.
print(f"Actual first param :{model.layer1.weight.data[0][0]}")
print(f"Predicting first param: {model.predict_param_by_idx(0)}")
# %%

# after regen.
new_weights = model.predict_own_weigths(shift_by=0.05)
model.set_weights(new_weights)
print(f"First param after regeneration: {model.layer1.weight.data[0][0]}")
# %%
