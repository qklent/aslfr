import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MyMultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            out_dim,
            qk_dim,
            v_dim,
            num_head,
            kernel_size=2,
            stride=2
        ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.num_head  = num_head
        self.qk_dim = qk_dim
        self.v_dim  = v_dim

        self.q = nn.Conv1d(embed_dim, qk_dim*num_head,kernel_size, stride)
        self.k = nn.Conv1d(embed_dim, qk_dim*num_head,kernel_size, stride) # stride=2 for token reduction, kernel>1 for mixing
        self.v = nn.Conv1d(embed_dim, v_dim*num_head,kernel_size, stride)

        self.out = nn.Conv1d(v_dim*num_head, out_dim, 1)
        self.scale = 1/(qk_dim**0.5)

    #https://github.com/pytorch/pytorch/issues/40497
    def forward(self, x):
        B,dim,L= x.shape

        num_head = self.num_head
        qk_dim = self.qk_dim
        v_dim = self.v_dim

        q = self.q(x) #B,qk_dim,L
        k = self.k(x)
        v = self.v(x)
        # B, N, L, Q
        q = q.reshape(B, num_head, qk_dim//self.kernel_size, L).permute(0,1,3,2).contiguous()
        k = k.reshape(B, num_head, qk_dim//self.kernel_size, L)#.permute(0,1,2,3).contiguous()
        v = v.reshape(B, num_head, v_dim//self.kernel_size,  L).permute(0,1,3,2).contiguous()

        dot = torch.matmul(q, k) * self.scale  # H L L
        attn = F.softmax(dot, -1)    # L L

        v = torch.matmul(attn, v)  # L H dim
        v = v.permute(0,1,3,2).reshape(B, v_dim*num_head,L).contiguous()
        out = self.out(v)
        return out

#---
embed_dim = 512
out_dim   = 512
qk_dim    = 512//4 #for one head
v_dim     = 512//4
num_head  = 4
max_length = 96
batch_size = 4

mha = MyMultiHeadAttention(
    embed_dim,
    out_dim,
    qk_dim,
    v_dim,
    num_head,
)
x  = torch.from_numpy(np.random.uniform(-1,1,(batch_size, embed_dim, max_length))).float()

y=mha(x)
print(y.shape) #torch.Size([4, 512, 96])
