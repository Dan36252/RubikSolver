from Code.Data.CubeState import CubeState
from Code.Data.TransformerDataIO import load_f2l_solver_data
from Code.Solver.CustomTransformerModel import multihead_attention
import random, torch

x = torch.randint(1, 20, (3, 73, 512), dtype=torch.float32)
print(x)
print(x.shape)
y = multihead_attention(512, 8).forward(x, masked=True)
print("-----------------------------------------------------------------------------")
print(y)
print(y.shape)
#load_f2l_solver_data()