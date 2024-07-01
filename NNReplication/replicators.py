# %%
import torch.nn as nn
import torch
from tqdm import tqdm

class RegenerationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_params = None

    def predict_own_weights(self, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError

    def predict_param_by_idx(self, idx):
        raise NotImplementedError

class SimpleModel(RegenerationBase):
    def __init__(self, layers_size=100,bias=True,act="ReLU"):
        super().__init__()
        self.activation = getattr(nn,act)()
        self.layer1 = nn.Linear(layers_size, layers_size, bias=bias)
        self.layer2 = nn.Linear(layers_size, 1, bias=bias)

        self.target_params = []
        for pmatrix in self.parameters():
            for p in pmatrix.view(-1):
                self.target_params.append(p.data)

        projection = nn.Linear(
            layers_size, len(self.target_params)
        ).weight.data.requires_grad_(False)
        self.register_buffer("projection", projection)

    def forward(self, x):
        for p in self.parameters():
            if p.dtype == torch.float16:
                x = x.half()
            break

        x = x @ self.projection
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x
            
    @torch.no_grad()
    def predict_own_weights(self, shift_by=0, shrink=1, batch_size=32):
        num_params = len(self.target_params)
        num_batches = (num_params + batch_size - 1) // batch_size

        
        max_batch_size = min(batch_size, num_params)
        batch_one_hot = torch.zeros((max_batch_size, num_params), device=self.projection.device)
        indices = torch.arange(max_batch_size, device=self.projection.device)

        all_weights = []

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_params)

            current_batch_size = end_idx - start_idx
            if current_batch_size != max_batch_size:
                batch_one_hot = batch_one_hot[:current_batch_size, :]
                indices = indices[:current_batch_size]

            batch_one_hot.zero_()
            batch_one_hot[torch.arange(current_batch_size), start_idx + indices] = 1

            batch_weights = self.forward(batch_one_hot) / shrink + shift_by
            all_weights.append(batch_weights)

        weights = torch.cat(all_weights, dim=0).squeeze().tolist()

        return weights

    @torch.no_grad()
    def set_weights(self, weights,shrink=1):
        if len(weights) != len(self.target_params):
            raise ValueError(
                "length of weights must match length of self.target_params"
            )

        i = 0
        for param in self.parameters():
            flat_param = param.view(-1)
            for p in range(flat_param.size(0)):
                weight = weights[i]
                weight = weight / shrink
                flat_param[p] = weight
                self.target_params[i] = torch.tensor(weight)
                i += 1
        assert i == len(self.target_params)

    def predict_param_by_idx(self, idx):
        num_params = len(self.target_params)
        one_hot_coordinate = torch.eye(num_params)[idx].to(self.projection.device)
        return self.forward(one_hot_coordinate)
    
    def reinitialize_model(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


# %%

if __name__ == "__main__":
    model = SimpleModel(150)
    from time import time
    s = time()
    new_weights = model.predict_own_weights(shift_by=0.1)
    print(time() - s)
    print(new_weights[:3])

    s = time()
    new_weights_2 = model.predict_own_weights(shift_by=0.1)
    print(time() - s)
    print(new_weights_2[:3])
# %%