# %%
# %load_ext autoreload
# %autoreload 2
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from typing import List
import torch
from torch import Tensor
from NNReplication import replicators
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
from config import DEVICE
import scipy.stats

def plot_corr(model: replicators.RegenerationBase,shift_by,shrink=1,name=""):
    # params = model.target_params
    params = []
    for pmatrix in model.parameters():
        for p in pmatrix.view(-1):
            params.append(p.data.clone())

    predictions = model.predict_own_weights(shift_by,shrink)
    weights = [param.item() for param in params]
    
    correlation_coefficient, p_value = scipy.stats.pearsonr(predictions, weights)
    print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")
    print(f"P-value: {p_value}")

    plt.title(f'(coef:{correlation_coefficient:.3f})')
    plt.scatter(weights, predictions)
    plt.xlabel("Weight")
    plt.ylabel("Prediction")
    plt.grid(True)
    show_plot(plt,name)
    
def regeneration_process(model: replicators.RegenerationBase, regen_steps, shift_by=0.05,batch_size=32,shrink=1):
    norms = []
    for i in tqdm(range(regen_steps),leave=False):
        norm = torch.norm(torch.tensor(model.target_params,device=DEVICE), 1)
        norms.append(norm.item())
        new_weights = model.predict_own_weights(shift_by=shift_by,batch_size=batch_size)
        # plot_corr(model,shift_by) # might give an error if model norms blew up. 
        model.set_weights(new_weights,shrink)
    return norms

import os
SAVE = True
def show_plot(plt, filename):
    output_dir = 'plot_images'  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    if SAVE:
        plt.savefig(filepath)
    else:
        plt.show()  
    plt.close()  
from scipy.stats import entropy
def compute_entropy(weights):
    weights_shifted = weights + abs(np.min(weights))
    probability_distribution = weights_shifted / np.sum(weights_shifted)
    return entropy(probability_distribution)


def plot_family_of_networks(num_networks, regen_steps, shift_by,bias=True,layer_size=70,act="ReLU",shrink=1,batch_size=64):
    model_norms = []
    final_weights = []  
    mse_values = []  

    for _ in tqdm(range(num_networks)):
        model = replicators.SimpleModel(layers_size=layer_size,bias=bias,act=act).to(DEVICE)
        norms = regeneration_process(model, regen_steps, shift_by,batch_size=batch_size,shrink=shrink)
        if norms[-1] != np.inf and not np.isnan(norms[-1]):  # only models that don't blow up
            model_norms.append((model, norms))
            weights_vector = torch.nn.utils.parameters_to_vector(model.target_params).detach().clone().cpu().numpy()
            final_weights.append(weights_vector)
            prediction = np.array(model.predict_own_weights(shift_by))

            mse = ((weights_vector - prediction) ** 2).mean()
            mse_values.append(mse)

        plt.plot(norms, alpha=0.5)#, label=f'Model {i+1} (size: {layer_size})')
        
    
    print(f"Out of {num_networks} networks, {num_networks - len(final_weights)} blew up.")
    plt.title(f'Norms Trend for {num_networks} Networks (shift by: {shift_by})')
    plt.xlabel('Generation')
    plt.ylabel('L1 Norm')
    plt.legend()
    plt.grid(True)
    
    
    filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_norms_trends.png'
    show_plot(plt,filename)

    
    
    perplexity = 8
    if len(final_weights) < perplexity:
        print("many model belw up aborting tsne")
    
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=300)
        tsne_results = tsne.fit_transform(np.array(final_weights))
        print(norms)

        plt.figure(figsize=(10, 6))
        mse_values_to_plot = np.clip(mse_values,0,1000)
        mse_colors = mse_values_to_plot / np.max(mse_values_to_plot)  
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=mse_colors, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='MSE')
        plt.title('t-SNE Visualization of Network Weights with MSE Coloring')
        plt.grid(True)
    
        filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_tsne.png'
        show_plot(plt,filename)
    
    if len(final_weights) < 2:
        print("many models blew up aborting everythign")
        return []
    
    entropies = [compute_entropy(weights) for weights in final_weights]
    networks_data = list(zip(final_weights, mse_values, entropies))
    networks_data.sort(key=lambda x: x[2])
    lowest_entropy_networks = networks_data[:2]
    highest_entropy_networks = networks_data[-2:]

    def plot_distribution(networks, title,name):
        fig, axes = plt.subplots(1, len(networks), figsize=(12, 6), sharey=True)
        fig.suptitle(title)

        for ax, (weights, mse, ent) in zip(axes, networks):
            ax.hist(weights, bins=30, alpha=0.7)
            ax.set_title(f'Entropy: {ent:.2f}\nMSE: {mse:.2f}')
            ax.set_xlabel('Weights')
            if ax == axes[0]:
                ax.set_ylabel('Frequency')

        plt.tight_layout()

        filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_{name}_plot_dist.png'
        show_plot(plt,filename)

        
    plot_distribution(lowest_entropy_networks, 'Distribution of Two Networks with Lowest Entropy', 'lowest_entropy_distribution')
    plot_distribution(highest_entropy_networks, 'Distribution of Two Networks with Highest Entropy', 'highest_entropy_distribution')


    print(f"Out of {num_networks} networks, {num_networks - len(final_weights)} blew up.")
    
    
    networks_data = list(zip(model_norms, mse_values, entropies))
    return networks_data


SAVE=True
shift_by = 0.001
shrink = 1
num_networks=20
regen_steps=5
layer_size=316
act="ReLU"
bias=False
batch_size = 2**4

networks_data = plot_family_of_networks(num_networks=num_networks, regen_steps=regen_steps, shift_by=shift_by,layer_size=layer_size,act=act,bias=bias,shrink=shrink,batch_size=batch_size)
networks_sorted_by_entropy = sorted(networks_data, key=lambda x: x[2])
networks_sorted_by_mse = sorted(networks_data, key=lambda x: x[1])

net = networks_sorted_by_mse[0]
filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_best_mse_corr.png'
plot_corr(net[0][0],shift_by,name=filename)
print("MSE: ", net[1])
print("Entropy: ", net[2])
plt.hist(torch.vstack(net[0][0].target_params).numpy())

filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_best_mse_dist.png'
show_plot(plt,filename)
net = networks_sorted_by_mse[-1]
filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_worst_mse_corr.png'
plot_corr(net[0][0],shift_by,name=filename)
print("MSE: ", net[1])
print("Entropy: ", net[2])
plt.hist(torch.vstack(net[0][0].target_params).numpy())

filename = f'{num_networks}_networks_shift_{shift_by}_size_{layer_size}_act_{act}_worst_mse_dist.png'
show_plot(plt,filename)

# %%

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def regeneration_process_noise(model: replicators.RegenerationBase, regen_steps, shift_by=0.05,batch_size=32,shrink=1):
    norms = []
    for i in tqdm(range(regen_steps),leave=False):
        norm = torch.norm(torch.tensor(model.target_params,device=DEVICE), 1)
        norms.append(norm.item())
        new_weights = model.predict_own_weights(shift_by=shift_by,batch_size=batch_size)
        # plot_corr(model,shift_by) # might give an error if model norms blew up. 
        if i == regen_steps - 1:
            model.set_weights(new_weights,shrink)
            break
            
        # new_weights[0] = 0.2
        # new_weights[1] = 0.5
        # new_weights[2] = 0.3
        # new_weights[1] = 1
        model.set_weights(new_weights,shrink)
    return norms
shift_by = 0.001
shrink = 1
layers_size=316
bias=True
act="ReLU"
batch = 2**4
model_1 = replicators.SimpleModel(layers_size=layers_size,bias=bias,act=act).to(DEVICE)
norms_1 = regeneration_process_noise(model_1, regen_steps=5,shift_by=shift_by,shrink=shrink,batch_size=batch)

SAVE = False

plt.figure(figsize=(10, 6))
plt.plot(norms_1, label=f'Model 1 (size:{len(model_1.target_params)})', marker='o')
plt.title(f'Norms Trend for Model 1 and Model 2, shift_by={shift_by}')
plt.xlabel('Generation')
plt.ylabel('L1 Norm')
plt.legend()
plt.grid(True)
filename = f'networks_shift_{shift_by}_size_{layer_size}_act_{act}_norms_with_noise.png'
show_plot(plt,filename)
# %%

filename = f'networks_shift_{shift_by}_size_{layers_size}_act_{act}_corr_with_noise.png'
plot_corr(model_1,shift_by=shift_by,name=filename)

plt.hist(torch.vstack(model_1.target_params).numpy())
filename = f'networks_shift_{shift_by}_size_{layer_size}_act_{act}_dist_with_noise.png'
show_plot(plt,filename)

print(model_1.target_params[0:40])
idx = -10
print((model_1.predict_param_by_idx(idx) + shift_by)/shrink, "=?", model_1.target_params[idx])
print(model_1.predict_param_by_idx(idx))
# %%

# %%

# %%
