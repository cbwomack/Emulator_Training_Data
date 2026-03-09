#%% [1] IMPORTS AND MOCK DATA GENERATION
import jax
import jax.numpy as jnp
import numpy as np
import optax
import utils_inverse

agents = utils_inverse.AGENTS_DEFAULT  # ("CO2", "CH4", "N2O", "Sulfur", "BC")
T = 144       # Time steps
output_dim = 46  # Dimension of the vector target (e.g., spatial bins)
scenarios = ["ssp126", "ssp370"]
lat_coords = np.linspace(-88, 88, output_dim)

n_agents = len(agents)
emis_dict_mock = {}
for scen in scenarios:
    emis_dict_mock[scen] = np.random.rand(n_agents, T).astype(np.float32)

# Create Mock Targets Dictionary (Vector)
targets_dict_mock = {}
for scen in scenarios:
    targets_dict_mock[scen] = np.random.rand(T, output_dim).astype(np.float32)

print("Setup complete. Mock data generated.")

#%% [2] Verify input/output shapes

key = jax.random.PRNGKey(42)
input_dim = 25 # Arbitrary input size (e.g., 5 agents * 5 features)
hidden_sizes = [16, 16]

# A. Test Initialization
params = utils_inverse.init_mlp_params_vector(key, input_dim, hidden_sizes, output_dim)
print(f"1. Init Params: {len(params)} layers created.")

# Check shapes of the last layer
last_w = params[-1]['W']
last_b = params[-1]['b']
assert last_w.shape == (hidden_sizes[-1], output_dim), f"Last layer W shape mismatch: {last_w.shape}"
assert last_b.shape == (output_dim,), f"Last layer b shape mismatch: {last_b.shape}"
print("   - Shapes verified.")

# B. Test Forward Pass
dummy_input = jnp.ones((5, input_dim)) # Batch of 5
output = utils_inverse.mlp_forward_vector(params, dummy_input)
assert output.shape == (5, output_dim), f"Output shape mismatch: {output.shape}"
print(f"2. Forward Pass: Output shape {output.shape} matches expected (Batch, OutDim).")

#%% [3] Test emulator training

# Create dummy training data
Xtr = jax.random.normal(key, (100, input_dim))
ytr = jax.random.normal(key, (100, output_dim))

# Run training
params_trained, losses = utils_inverse.train_mlp_sgd_vector(
    params, Xtr, ytr, K=10, lr=1e-2
)

print(f"3. Training Loop: Final loss {losses[-1]:.4f}")
assert not jnp.isnan(losses[-1]), "Training resulted in NaN loss."


#%% [4] Test data construction

dataset = utils_inverse.build_dataset_vector_targets(
    emis_dict_mock,
    targets_dict_mock,
    scenarios,
    agents=agents
)

print(f"4. Build Dataset: Processed {len(dataset)} scenarios.")
X_ex, y_ex, name_ex = dataset[0]

# Check alignment
# X features depends on 'make_features_emissions_generic' (usually 5 features per agent)
expected_feat_dim = len(agents) * 5
assert X_ex.shape[1] == expected_feat_dim, f"Feature dim {X_ex.shape[1]} != {expected_feat_dim}"
assert y_ex.shape[1] == output_dim, f"Target dim {y_ex.shape[1]} != {output_dim}"
assert X_ex.shape[0] == y_ex.shape[0], "Time dimension mismatch between X and y."
print(f"   - Scenario '{name_ex}': X shape {X_ex.shape}, y shape {y_ex.shape}.")

#%% [5] Test full generation and evaluation of baseline emulator

# Setup "Train" and "Eval" sets (using same mock data for simplicity)
eval_emis_sets = {"Tier 1": emis_dict_mock}
eval_targets_sets = {"Tier 1": targets_dict_mock}

results, preds, truths = utils_inverse.generate_and_eval_baseline_emulator_vector(
    emis_dict_train=emis_dict_mock,
    targets_dict_train=targets_dict_mock,
    eval_emis_sets=eval_emis_sets,
    eval_targets_sets=eval_targets_sets,
    output_dim=output_dim,
    lat_coords=lat_coords,
    hidden_sizes=[16],
    K=50,
    verbose=True
)

# 4. Verify Outputs
print("\n5. Verification:")
t1_res = results['Tier 1']

# Check Structure
print(f"   - Keys in Result: {t1_res.keys()}")
print(f"   - Keys in 'mean': {t1_res['mean'].keys()}")

# Check Values
global_mean = t1_res['mean']['global']
zonal_mean = t1_res['mean']['zonal']

print(f"   - Global Mean NRMSE: {global_mean:.4f}")
print(f"   - Zonal Mean NRMSE Shape: {zonal_mean.shape}")

# Assertions
assert isinstance(global_mean, float), "Global mean should be a float"
assert zonal_mean.shape == (output_dim,), f"Zonal mean should be shape ({output_dim},)"
assert 'global' in t1_res[scenarios[0]], "Individual scenarios should have 'global' key"
assert 'zonal' in t1_res[scenarios[0]], "Individual scenarios should have 'zonal' key"

print("   - All assertions passed.")
# %%
