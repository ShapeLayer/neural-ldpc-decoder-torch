import torch
import numpy as np
from boosted_neural_ldpc_decoder.ConnectingMatrix import ConnectingMatrix


class ConnectingMatrixTorch:
    def __init__(
            self,
            connecting_matrix: ConnectingMatrix,
            device: torch.device = torch.device('cpu'),
            dtype_w_odd2even: torch.dtype = torch.float32,
            dtype_w_skipconn2even: torch.dtype = torch.float32,
            dtype_w_even2odd: torch.dtype = torch.float32,
            dtype_w_even2odd_with_self: torch.dtype = torch.float32,
            dtype_w_output: torch.dtype = torch.float32,
            dtype_w_skipconn2odd: torch.dtype = torch.float32,
            dtype_lifting_matrix: torch.dtype = torch.float32,
    ):
        self.device = device

        self.N = connecting_matrix.N
        self.M = connecting_matrix.M
        self.Z = connecting_matrix.Z

        self.basegraph = connecting_matrix.basegraph.copy()
        self.sum_edge_c = connecting_matrix.sum_edge_c.copy()
        self.sum_edge_v = connecting_matrix.sum_edge_v.copy()
        self.sum_edge = connecting_matrix.sum_edge.copy()

        self.dtype_w_odd2even = dtype_w_odd2even
        self.dtype_w_skipconn2even = dtype_w_skipconn2even
        self.dtype_w_even2odd = dtype_w_even2odd
        self.dtype_w_even2odd_with_self = dtype_w_even2odd_with_self
        self.dtype_w_output = dtype_w_output
        self.dtype_w_skipconn2odd = dtype_w_skipconn2odd
        self.dtype_lifting_matrix = dtype_lifting_matrix

        self.neurons_per_even_layer = connecting_matrix.neurons_per_even_layer
        self.neurons_per_odd_layer = connecting_matrix.neurons_per_odd_layer

        self.W_odd2even = torch.tensor(connecting_matrix.W_odd2even, dtype=self.dtype_w_odd2even, device=self.device)
        self.W_skipconn2even = torch.tensor(connecting_matrix.W_skipconn2even, dtype=self.dtype_w_skipconn2even,
                                            device=self.device)
        self.W_even2odd = torch.tensor(connecting_matrix.W_even2odd, dtype=self.dtype_w_even2odd, device=self.device)
        self.W_even2odd_with_self = torch.tensor(connecting_matrix.W_even2odd_with_self,
                                                 dtype=self.dtype_w_even2odd_with_self, device=self.device)
        self.W_output = torch.tensor(connecting_matrix.W_output, dtype=self.dtype_w_output, device=self.device)
        self.W_skipconn2odd = torch.tensor(connecting_matrix.W_skipconn2odd, dtype=self.dtype_w_skipconn2odd,
                                           device=self.device)

        self.lifting_matrix_1 = torch.tensor(connecting_matrix.lifting_matrix_1, dtype=self.dtype_lifting_matrix,
                                             device=self.device)
        self.lifting_matrix_2 = torch.tensor(connecting_matrix.lifting_matrix_2, dtype=self.dtype_lifting_matrix,
                                             device=self.device)
