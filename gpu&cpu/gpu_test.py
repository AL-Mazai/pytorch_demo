import torch

print("hello")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
