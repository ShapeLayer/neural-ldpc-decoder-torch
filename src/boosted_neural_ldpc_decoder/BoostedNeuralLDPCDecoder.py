from typing import Optional

import torch
import torch.nn as nn

from boosted_neural_ldpc_decoder.struct.Clipping import Clipping
from boosted_neural_ldpc_decoder.ConnectingMatrixTorch import ConnectingMatrixTorch
from boosted_neural_ldpc_decoder.struct.DecoderType import DecoderType
from boosted_neural_ldpc_decoder.struct.NodeType import NodeType
from boosted_neural_ldpc_decoder.struct.NodeWeightSharingConfig import NodeWeightSharingConfig
from boosted_neural_ldpc_decoder.struct.ParamType import ParamType


class BoostedNeuralLDPCDecoder(nn.Module):
    def __init__(
            self,
            iter_node_counts,
            batch_size,
            connecting_matrix: ConnectingMatrixTorch,
            node_weight_sharing_config: NodeWeightSharingConfig = NodeWeightSharingConfig(
                cn_weight_sharing=3,
                ucn_weight_sharing=0,
                vn_weight_sharing=0,
            ),

            decoding_type: DecoderType = DecoderType.QMS,
            decoder_qms_qbit: int = 5,
            fixed_iterative_nodes: list[int] = [],
            # converted from fixed_iter
            # fixed_iter directions fixed range(0, fixed_iter)
            # but fixed_iterative_nodes is list of int

            fixed_iterative_nodes_init_weight: int = 0,
            allowed_weight_range: Clipping = Clipping(start=0, end=2),
            allowed_bias_range: Clipping = Clipping(start=0, end=2),
            allowed_llr_range: Clipping = Clipping(abs=20.0),
            dtype_cn_weight: torch.dtype = torch.float32,
            dtype_ucn_weight: torch.dtype = torch.float32,
            dtype_vn_weight: torch.dtype = torch.float32,
            init_cn_weight: float = 1,
            init_ucn_weight: float = 1,
            init_vn_weight: float = 1,
            dtype_cn_bias: torch.dtype = torch.float32,
            dtype_ucn_bias: torch.dtype = torch.float32,
            dtype_vn_bias: torch.dtype = torch.float32,
            init_cn_bias: float = 1,
            init_ucn_bias: float = 1,
            init_vn_bias: float = 1,
    ):
        super(BoostedNeuralLDPCDecoder, self).__init__()
        self.iter_node_counts = iter_node_counts
        self.batch_size = batch_size

        self.conn_mat = connecting_matrix

        self.N = self.conn_mat.N
        self.M = self.conn_mat.M
        self.Z = self.conn_mat.Z
        self.sum_edge = self.conn_mat.sum_edge
        self.neurons_per_odd_layer = self.conn_mat.neurons_per_odd_layer
        self.neurons_per_even_layer = self.conn_mat.neurons_per_even_layer

        self.node_weight_sharing_config = node_weight_sharing_config
        self.decoding_type = decoding_type
        self.decoder_qms_qbit = decoder_qms_qbit
        self.fixed_iterative_nodes = fixed_iterative_nodes
        self.fixed_iterative_nodes_init_weight = fixed_iterative_nodes_init_weight
        self.allowed_weight_range = allowed_weight_range
        self.allowed_bias_range = allowed_bias_range
        self.allowed_llr_range = allowed_llr_range

        self.dtype_cn_weight = dtype_cn_weight
        self.dtype_ucn_weight = dtype_ucn_weight
        self.dtype_vn_weight = dtype_vn_weight
        self.init_cn_weight = init_cn_weight
        self.init_ucn_weight = init_ucn_weight
        self.init_vn_weight = init_vn_weight
        self.dtype_cn_bias = dtype_cn_bias
        self.dtype_ucn_bias = dtype_ucn_bias
        self.dtype_vn_bias = dtype_vn_bias
        self.init_cn_bias = init_cn_bias
        self.init_ucn_bias = init_ucn_bias
        self.init_vn_bias = init_vn_bias

        self.register_buffer('W_odd2even', self.conn_mat.W_odd2even)
        self.register_buffer('W_skipconn2even', self.conn_mat.W_skipconn2even)
        self.register_buffer('W_even2odd', self.conn_mat.W_even2odd)
        self.register_buffer('W_even2odd_with_self', self.conn_mat.W_even2odd_with_self)
        self.register_buffer('W_output', self.conn_mat.W_output)
        self.register_buffer('W_skipconn2odd', self.conn_mat.W_skipconn2odd)
        self.register_buffer('Lift_Matrix1', self.conn_mat.lifting_matrix_1)
        self.register_buffer('Lift_Matrix2', self.conn_mat.lifting_matrix_2)

        self._register_params()

    def _param_name(self, param_type: ParamType, node_type: NodeType, iterative_node_identifier: int):
        return f"{param_type.value}_{node_type.value}_{iterative_node_identifier}"

    def _register_params(self):
        for node_type, sharing_type in self.node_weight_sharing_config:
            if sharing_type == 0:
                continue

            # Determine parameter shape based on sharing type
            if sharing_type in [1, 4]:  # Per-edge weights
                param_shape = (self.sum_edge,)
            elif sharing_type in [2, 5]:  # Per-node weights
                if node_type in [NodeType.CN, NodeType.UCN]:
                    param_shape = (self.M,)
                else:  # VN
                    param_shape = (self.N,)
            elif sharing_type == 3:  # Single weight
                param_shape = (1,)
            else:
                raise ValueError(f"Unsupported sharing type {sharing_type} for {node_type}")

            param_value = 0
            param_dtype = None
            if node_type == NodeType.CN:
                param_value = self.init_cn_weight
                param_dtype = self.dtype_cn_weight
            elif node_type == NodeType.UCN:
                param_value = self.init_ucn_weight
                param_dtype = self.dtype_ucn_weight
            elif node_type == NodeType.VN:
                param_value = self.init_vn_weight
                param_dtype = self.dtype_vn_weight

            iterations_to_create = None
            if sharing_type in [1, 2, 3]:  # Independent weights per iteration
                iterations_to_create = [i for i in range(self.iter_node_counts)]
            else:
                iterations_to_create = [self.iter_node_counts]

            for iteration in iterations_to_create:
                weight_name = self._param_name(ParamType.Weight, node_type, iteration)
                init_tensor = torch.full(param_shape, param_value, dtype=param_dtype)
                param = nn.Parameter(init_tensor)
                setattr(self, weight_name, param)

    def _apply_constraints(self):
        for node_type, sharing_type in self.node_weight_sharing_config:
            if sharing_type == 0: continue
            iterations_to_create = None
            if sharing_type in [1, 2, 3]:  # Independent weights per iteration
                iterations_to_create = [i for i in range(self.iter_node_counts)]
            else:
                iterations_to_create = [self.iter_node_counts]
            for iteration in iterations_to_create:
                for param_type in [ParamType.Weight, ParamType.Bias]:
                    param_name = self._param_name(param_type, node_type, iteration)
                    param = getattr(self, param_name)
                    if param is None:
                        continue
                    if param_type == ParamType.Weight:
                        param.data.clamp_(self.allowed_weight_range.start, self.allowed_weight_range.end)
                    elif param_type == ParamType.Bias:
                        param.data.clamp_(self.allowed_bias_range.start, self.allowed_bias_range.end)

    def _get_param(self, param_type: ParamType, node_type: NodeType, iterative_node_identifier: int):
        name = self._param_name(param_type, node_type, iterative_node_identifier)
        if hasattr(self, name):
            return getattr(self, name)
        return None

    def _quantize_message(self, x: torch.Tensor, q_bit: int) -> torch.Tensor:
        """Quantize message for QMS decoder."""
        q_step = 2.0 ** (1 - q_bit)
        q_range = 2.0 ** q_bit - 1

        sign = torch.sign(x)
        abs_val = torch.abs(x)

        # Quantize the absolute value
        quant_abs = torch.round(abs_val / q_step) * q_step
        # Clip to quantization range
        quant_abs = torch.clamp(quant_abs, 0.0, q_range * q_step)

        # Apply sign back
        return quant_abs * sign

    def fetch_param(self, param_type: ParamType, node_type: NodeType, curr_iter: int) -> Optional[torch.Tensor]:
        """Fetch the appropriate weight for node type and current iteration."""
        sharing_type = self.node_weight_sharing_config.get(node_type)
        if sharing_type == 0:
            return None

        if sharing_type in [1, 2, 3]:
            # Independent weights per iteration
            return self._get_param(param_type, node_type, curr_iter)
        elif sharing_type in [4, 5]:
            # Fixed weights - use weights from fixed iterations
            if self.fixed_iterative_nodes:
                # Find the closest fixed iteration that's less than or equal to curr_iter
                valid_iters = [i for i in self.fixed_iterative_nodes if i <= curr_iter]
                if valid_iters:
                    iter_idx = max(valid_iters)
                    return self._get_param(param_type, node_type, iter_idx)
                # If no valid fixed iteration, use the first one
                return self._get_param(param_type, node_type, self.fixed_iterative_nodes[0])
            # If no fixed iterations specified, use iteration 0
            return self._get_param(param_type, node_type, 0)
        return None

    def get_trainable_parameters(self):
        """Return all trainable parameters for optimization."""
        params = []
        for node_type, sharing_type in self.node_weight_sharing_config:
            if sharing_type == 0:
                continue

            iterations_to_consider = None
            if sharing_type in [1, 2, 3]:
                iterations_to_consider = range(self.iter_node_counts)
            else:  # sharing_type in [4, 5]
                iterations_to_consider = self.fixed_iterative_nodes if self.fixed_iterative_nodes else [0]

            for iter_idx in iterations_to_consider:
                if iter_idx < self.fixed_iterative_nodes_init_weight:
                    continue

                param = self._get_param(ParamType.Weight, node_type, iter_idx)
                if param is not None:
                    params.append(param)
        return params

    def forward(self, xa):
        batch_size = xa.shape[0]
        xa_input = xa.transpose(1, 2)  # [batch, Z, N]

        # Initialize LLR
        prev_llr = torch.zeros((batch_size, self.Z, self.sum_edge),
                               dtype=torch.float32, device=self.conn_mat.device)

        outputs = []

        for curr_iter in range(self.iter_node_counts):
            vn_weight = self.fetch_param(ParamType.Weight, NodeType.VN, curr_iter)
            # Apply VN weights if specified
            vn_weight = self.fetch_param(ParamType.Weight, NodeType.VN, curr_iter)
            if vn_weight is not None:
                if vn_weight.numel() == 1:
                    weighted_input = xa_input * vn_weight
                elif vn_weight.numel() == self.N:
                    weighted_input = xa_input * vn_weight.view(1, 1, -1)
                else:
                    weighted_input = xa_input
            else:
                weighted_input = xa_input

            # Variable node update
            x0 = torch.matmul(weighted_input, self.W_skipconn2even)
            x1 = torch.matmul(prev_llr, self.W_odd2even)
            x2 = x0 + x1

            # in case of QMS
            if self.decoding_type == DecoderType.QMS:
                x2 = self._quantize_message(x2, self.decoder_qms_qbit)
            else:
                x2 = torch.clamp(x2, self.allowed_llr_range.start, self.allowed_llr_range.end)

            # transform using lifting_matrix
            x2 = x2.transpose(1, 2)  # [batch, sum_edge, Z]
            x2 = x2.reshape(batch_size, self.neurons_per_odd_layer * self.Z)
            x2 = torch.matmul(x2, self.conn_mat.lifting_matrix_1.t())
            x2 = x2.reshape(batch_size, self.neurons_per_odd_layer, self.Z)
            x2 = x2.transpose(1, 2)  # [batch, Z, neurons_per_odd_layer]

            # Tile for check node computation
            x_tile = x2.repeat(1, 1, self.neurons_per_odd_layer)
            W_input_reshape = self.W_even2odd.t().reshape(-1)
            #   TODO: caching transpose option

            # Check node update
            x_tile_mul = x_tile * W_input_reshape
            x2_1 = x_tile_mul.reshape(batch_size, self.Z, self.neurons_per_odd_layer, self.neurons_per_odd_layer)

            if self.decoding_type == DecoderType.SP:
                # Sum-product operations
                x2_tanh = torch.tanh(0.5 * x2_1)
                mask = (x2_tanh.abs() > 0).float()
                x2_abs = x2_tanh * mask + (1.0 - mask)  # Avoid NaNs in product
                x3 = torch.prod(x2_abs, dim=3)
                x3_clipped = torch.clamp(x3, -0.9999, 0.9999)  # Prevent numerical issues
                x_output_0 = 2.0 * torch.atanh(x3_clipped)
            else:
                # Min-sum operations
                x2_abs = torch.abs(x2_1) + 10000 * (1 - (torch.abs(x2_1) > 0).float())
                x3 = torch.min(x2_abs, dim=3)[0]

                x2_2 = -x2_1
                x4 = torch.ones_like(x2_2) - 2 * (x2_2 < 0).float()
                x4_prod = -torch.prod(x4, dim=3)
                x_output_0 = x3 * torch.sign(x4_prod)

            # Get check node sign information for UCN weights
            cn_sign = torch.sign(x_output_0)
            ucn_mask = (cn_sign < 0).float()
            cn_mask = 1.0 - ucn_mask

            # Apply CN and UCN weights
            cn_weight = self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter)
            ucn_weight = self.fetch_param(ParamType.Weight, NodeType.UCN, curr_iter)

            # Process the absolute value for weight application
            x_output_abs = torch.abs(x_output_0)

            # Apply appropriate weights based on configuration
            if cn_weight is not None:
                if cn_weight.numel() == 1:
                    cn_weighted = x_output_abs * cn_weight
                elif cn_weight.numel() == self.sum_edge:
                    cn_weighted = x_output_abs * cn_weight.view(1, 1, -1)
                elif cn_weight.numel() == self.M:
                    # Transform M-length weight to sum_edge-length
                    expanded_weight = torch.matmul(cn_weight, self.W_skipconn2odd)
                    cn_weighted = x_output_abs * expanded_weight.view(1, 1, -1)
                else:
                    cn_weighted = x_output_abs
            else:
                cn_weighted = x_output_abs

            if ucn_weight is not None:
                if ucn_weight.numel() == 1:
                    ucn_weighted = x_output_abs * ucn_weight
                elif ucn_weight.numel() == self.sum_edge:
                    ucn_weighted = x_output_abs * ucn_weight.view(1, 1, -1)
                elif ucn_weight.numel() == self.M:
                    # Transform M-length weight to sum_edge-length
                    expanded_weight = torch.matmul(ucn_weight, self.W_skipconn2odd)
                    ucn_weighted = x_output_abs * expanded_weight.view(1, 1, -1)
                else:
                    ucn_weighted = x_output_abs

                # Apply UCN weights to unsatisfied check nodes
                x_output_weighted = cn_weighted * cn_mask + ucn_weighted * ucn_mask
            else:
                x_output_weighted = cn_weighted

            # ReLU and sign application
            x_output_final = torch.relu(x_output_weighted)
            if self.decoding_type == DecoderType.QMS:
                x_output_final = self._quantize_message(x_output_final, self.decoder_qms_qbit)

            # Apply sign back
            next_llr = x_output_final * cn_sign
            prev_llr = next_llr  # Store for next iteration

            # Apply clipping if needed
            if self.allowed_llr_range.start is not None and self.allowed_llr_range.end is not None:
                next_llr = torch.clamp(next_llr, self.allowed_llr_range.start, self.allowed_llr_range.end)

            # Generate output for this iteration
            y_output = torch.matmul(next_llr, self.W_output)
            y_output = y_output.transpose(1, 2)  # [batch, N, Z]
            y_output = xa + y_output

            # Apply clipping to final output
            if self.allowed_llr_range.start is not None and self.allowed_llr_range.end is not None:
                y_output = torch.clamp(y_output, self.allowed_llr_range.start, self.allowed_llr_range.end)
            outputs.append(y_output.reshape(batch_size, self.N * self.Z))
        return outputs
