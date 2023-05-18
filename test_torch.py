import torch

print("Torch version:", torch.__version__)

print("CUDA device count:", torch.cuda.device_count())
#print("CUDA current device:", torch.cuda.current_device())
for devNumber in range(torch.cuda.device_count()):
    print(f"Device {devNumber}: {torch.cuda.get_device_name(devNumber)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
