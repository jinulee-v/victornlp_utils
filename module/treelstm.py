"""
@module treelstm

Implementation of Child-sum Tree-LSTM and its variants.
- Original
    Kai, T. S. et al.(2015) Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
- Edge-labeled version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChildSumTreeLSTM(nn.Module):
  """
  @class ChildSumTreeLSTM

  Implementation of Child-Sum Tree LSTM with VictorNLP dependency tree input.
  Dependency tree format:
  [
    {
      'dep': Integer,
      'head': Integer,
      'label': String
    },
    ...
  ]
  """

  def __init__(self, input_size, hidden_size):
    """
    Constructor for ChildSumTreeLSTM.
    
    @param self The object pointer.
    @param input_size Integer. Input size(word embedding).
    @param hidden_size Integer. Hidden size of theTreeLSTM unit.
    """
    super(ChildSumTreeLSTM, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.W_iou = nn.Linear(self.input_size, self.hidden_size * 3)
    self.W_f   = nn.Linear(self.input_size, self.hidden_size)
    self.U_iou = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
    self.U_f   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
  
  def _forward_node(self, j, children, x, h, c):
    """
    Constructor for ChildSumTreeLSTM.
    
    @param self The object pointer.
    @param j Integer. Index of current node.
    @param children LongTensor. Indices of the children of the current node.
    @param x Tensor(length, input_size). Input embedding of the current sentence.
    @param h Tensor(length, hidden_size). Hidden state of the current sentence.
    @param c Tensor(length, hidden_size). Cell state of the current sentence.
    """

    h_j_sum = torch.sum(torch.gather(h, 1, children))

    iou_j = self.W_iou(x[j]) + self.U_iou(h_j_sum)
    i_j, o_j, u_j = torch.split(iou_j, iou_j.size(1)//3, dim=1)
    i_j = F.sigmoid(i_j)
    o_j = F.sigmoid(o_j)
    u_j = F.tanh(u_j)

    c[j, :] = i_j * u_j
    for child in children:
      c[j, :] += (self.W_f(x[j]) + self.U_f(h[child])) * c[child]
    h[j, :] = o_j * F.tanh(c[j, :])
  
  def _forward_postorder(self, j, children_list, x, h, c, device):
    """
    Postorder traversal of the given tree.
    """
    children = children_list[j]
    if not children:
      return
    
    for child in children:
      self._forward_postorder(self, child, children_list, x, h, c)
    
    self._forward_node(j, torch.LongTensor(children).to(device), x, h, c)
  
  def forward(self, dp_tree, input_embeddings, root):
    """
    forward() override for nn.Module.

    @param dp_tree VictorNLP depednency tree format
    @param input_embeddings Tensor(length, input_size).
    @param root Integer. Starting point for _forward_postorder
    """
    device = input_embeddings.device

    # Create children_list
    children_list = []
    for arc in reversed(dp_tree):
      if len(children_list) < max(arc['dep'], arc['head']):
        children_list.extend([[] for _ in range(max(arc['dep'], arc['head'] - len(children_list)))])
      children_list[arrc['head']].append(arc['dep'])
    
    # Create empty hidden/cell state matrices
    h = torch.zeros(input_embeddings.size(0), self.hidden_size, device=device)
    c = torch.zeros_like(h)

    # Recursive call
    _forward_postorder(root, children_list, input_embeddings, h, c)

    return h, c