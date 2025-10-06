import torch
import torch.nn as nn
from .ConnectingMatrixTorch import ConnectingMatrixTorch


class NeuralLDPCDecoder(nn.Module):
    def __init__(
            self,
            iter_node_counts,
            batch_size,
            connecting_matrix: ConnectingMatrixTorch,
    ):
        super(NeuralLDPCDecoder, self).__init__()
        self.iter_node_counts = iter_node_counts
        self.batch_size = batch_size

        self.conn_mat = connecting_matrix

        self.N = self.conn_mat.N
        self.M = self.conn_mat.M
        self.Z = self.conn_mat.Z
        self.sum_edge = self.conn_mat.sum_edge
        self.neurons_per_odd_layer = self.conn_mat.neurons_per_odd_layer
        self.neurons_per_even_layer = self.conn_mat.neurons_per_even_layer

        # Register matrices as buffers (non-trainable parameters)
        self.register_buffer('W_odd2even', self.conn_mat.W_odd2even)
        self.register_buffer('W_skipconn2even', self.conn_mat.W_skipconn2even)
        self.register_buffer('W_even2odd', self.conn_mat.W_even2odd)
        self.register_buffer('W_output', self.conn_mat.W_output)
        self.register_buffer('Lift_Matrix1', self.conn_mat.lifting_matrix_1)
        self.register_buffer('Lift_Matrix2', self.conn_mat.lifting_matrix_2)

        # Learnable parameters for each iteration
        self.weights_var = nn.ParameterList([
            nn.Parameter(0.5 * torch.ones(self.conn_mat.sum_edge, dtype=torch.float32))
            for _ in range(iter_node_counts)
        ])
        self.biases_var = nn.ParameterList([
            nn.Parameter(torch.zeros(self.conn_mat.sum_edge, dtype=torch.float32))
            for _ in range(iter_node_counts)
        ])

    def forward(self, xa):
        batch_size = xa.shape[0]
        xa_input = xa.transpose(1, 2)  # [batch, Z, N]

        # Initialize LLR
        llr = torch.zeros((batch_size, self.Z, self.sum_edge),
                          dtype=torch.float32, device=xa.device)

        outputs = []

        for i in range(self.iter_node_counts):
            # Variable node update
            x0 = torch.matmul(xa_input, self.conn_mat.W_skipconn2even)
            x1 = torch.matmul(llr, self.conn_mat.W_odd2even)
            x2 = x0 + x1
            x2 = x2.transpose(1, 2)  # [batch, sum_edge, Z]
            x2 = x2.reshape(batch_size, self.neurons_per_odd_layer * self.Z)
            x2 = torch.matmul(x2, self.conn_mat.lifting_matrix_1.t())
            x2 = x2.reshape(batch_size, self.neurons_per_odd_layer, self.Z)
            x2 = x2.transpose(1, 2)  # [batch, Z, neurons_per_odd_layer]

            # Tile for check node computation
            x_tile = x2.repeat(1, 1, self.conn_mat.neurons_per_odd_layer)
            W_input_reshape = self.conn_mat.W_even2odd.t().reshape(-1)

            # Check node update
            x_tile_mul = x_tile * W_input_reshape
            x2_1 = x_tile_mul.reshape(batch_size, self.Z, self.neurons_per_odd_layer, self.neurons_per_odd_layer)

            # Min-sum operations
            x2_abs = torch.abs(x2_1) + 10000 * (1 - (torch.abs(x2_1) > 0).float())
            x3 = torch.min(x2_abs, dim=3)[0]

            x2_2 = -x2_1
            x4 = torch.ones_like(x2_2) - 2 * (x2_2 < 0).float()
            x4_prod = -torch.prod(x4, dim=3)
            x_output_0 = x3 * torch.sign(x4_prod)

            x_output_0 = x_output_0.transpose(1, 2)  # [batch, neurons_per_odd_layer, Z]
            x_output_0 = x_output_0.reshape(batch_size, self.Z * self.neurons_per_odd_layer)
            x_output_0 = torch.matmul(x_output_0, self.conn_mat.lifting_matrix_2)
            x_output_0 = x_output_0.reshape(batch_size, self.neurons_per_odd_layer, self.Z)
            x_output_0 = x_output_0.transpose(1, 2)  # [batch, Z, neurons_per_odd_layer]

            # Add learnable parameters
            x_output_1 = torch.abs(x_output_0) * self.weights_var[i] + self.biases_var[i]
            x_output_1 = x_output_1 * (x_output_1 > 0).float()  # ReLU
            llr = x_output_1 * torch.sign(x_output_0)  # Update LLR

            # Output
            y_output_2 = torch.matmul(llr, self.conn_mat.W_output)
            y_output_3 = y_output_2.transpose(1, 2)  # [batch, N, Z]
            y_output_4 = xa + y_output_3
            ya_output = y_output_4.reshape(batch_size, self.N * self.Z)
            outputs.append(ya_output)

        return outputs
