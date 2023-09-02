from model import *

device = torch.cuda.is_available()
N, seq_length, C = 2, 128, 164
positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
print (positions.shape)