import torch
import torch.nn as nn
import deepspeed

# ---------
# Toy model
# ---------
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x).sum()


# ---------
# Init
# ---------
device = "cuda"
model = ToyModel().to(device)

config = {
    "train_batch_size": 1,
    "zero_optimization": {
        "stage": 3
    },
    "fp16": {
        "enabled": False
    }
}

engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config,
)

# ---------
# Input
# ---------
x = torch.randn(2, 8, device=device)

# ---------
# Define loss fn
# ---------
def loss_fn(_):
    return engine(x)

# ---------
# Trigger torch.func
# ---------
print("Running torch.func.grad_and_value...")

grad_fn = torch.func.grad_and_value(loss_fn)

# dummy param (required by API)
dummy = torch.tensor(0.0, device=device)

grad, val = grad_fn(dummy)

print("SUCCESS (no crash)")
print("Value:", val)