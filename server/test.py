import torch
import numpy

print("PyTorch Version:", torch.__version__)
print("NumPy Version:", numpy.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
