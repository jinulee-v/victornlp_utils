"""
@module gcn

Implementation of Graph Convolution Network.
> Kipf & Welling. Semi-supervised Classification with Graph Convolution Networks.
"""

import torch
import torch.nn as nn

class GCN(nn.Module):
  """
  @class GCN

  Implementation of a Graph Convolution Network.
  > Kipf & Welling. Semi-supervised Classification with Graph Convolution Networks.
  """

  def __init__(self, input_size, hidden_size, num_layers):
    """
    Constructor for GCN.
    
    @param self The object pointer.
    @param input_size Integer. Input size(word embedding/hidden state layer before).
    @param hidden_size Integer. Hidden size of theTreeLSTM unit.
    """
    super(GCN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.layers = nn.ParameterList(
      [nn.Parameter(torch.zeros(self.input_size, self.hidden_size))] +
      [nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers - 1)]
    )
    self.biases = nn.ParameterList(
      [nn.Parameter(torch.zeros(self.hidden_size)) for _ in range(self.num_layers)]
    )
  
  def forward(self, adj_matrix, input_state):
    """
    forward() override for nn.Module.

    @param adj_matrix Tensor(batch_size, length, length). Adjacency matrix for the graph.
    @param input_state Tensor(batch_size, length, input_dim) Input vectors.
    @return hidden_state Tensor(batch_size, length, hidden_state). Last layer hidden state of the GCN module.
    """
    batch_size = adj_matrix.size(0)
    device = next(self.parameters()).device

    adj_matrix = adj_matrix.detach()

    norm = torch.sum(adj_matrix, dim=2, keepdim=True) + 1
    
    state = input_state
    for layer, bias in zip(self.layers, self.biases):
      # state = ReLU( D_sqrt * adj_matrix * D_sqrt * state * layer)
      state = torch.relu(
        torch.matmul(
          torch.matmul(
            adj_matrix, # Tensor(batch_size, length, length)
            state
          ), # Tensor(batch_size, length, input/hidden_dim)
          layer
        ) # Tensor(batch_size, length, hidden_dim)
      ) + torch.matmul(state, layer) + bias
    
    return state