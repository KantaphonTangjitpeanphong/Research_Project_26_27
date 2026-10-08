

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))
#     print("CUDA version PyTorch was built with:", torch.version.cuda)

#     # Actually run something on the GPU, not just check availability
#     x = torch.rand(3, 3).cuda()
#     y = torch.rand(3, 3).cuda()
#     z = x @ y
#     print("Test tensor multiply on GPU succeeded:")
#     print(z)
# else:
#     print("No GPU detected — check your driver and reinstall the matching cuXXX PyTorch build.")
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
