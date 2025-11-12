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

        self.outputs = [
            torch.zeros((self.batch_size, self.N * self.Z), dtype=torch.float32, device=self.conn_mat.device)
            for _ in range(self.iter_node_counts)
        ]
        self.llr = [
            torch.zeros((self.batch_size, self.Z, self.sum_edge), dtype=torch.float32, device=self.conn_mat.device)
            for _ in range(self.iter_node_counts + 1)
        ]

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
        """Apply constraints to weights and biases (if they exist)."""
        for node_type, sharing_type in self.node_weight_sharing_config:
            if sharing_type == 0:
                continue
                
            iterations_to_create = None
            if sharing_type in [1, 2, 3]:
                iterations_to_create = [i for i in range(self.iter_node_counts)]
            else:
                iterations_to_create = [self.iter_node_counts]
                
            for iteration in iterations_to_create:
                for param_type in [ParamType.Weight, ParamType.Bias]:
                    param_name = self._param_name(param_type, node_type, iteration)
                    if not hasattr(self, param_name):
                        continue
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
        """Quantize message for QMS decoder - matches TF implementation."""
        if q_bit == 6:
            q_value = torch.clamp(torch.round(x), -15.5, 15.5)
        elif q_bit == 5:
            q_value = torch.clamp(torch.round(x * 2) / 2, -7.5, 7.5)
        elif q_bit == -5:
            q_value = torch.clamp(torch.round(x), -15, 15)
        elif q_bit == 4:
            q_value = torch.clamp(torch.round(x), -7, 7)
        elif q_bit == 3:
            q_value = torch.clamp(torch.round(x / 2) * 2, -6, 6)
        else:
            return x
        
        # Straight-through estimator: forward uses quantized, backward uses clipped
        if q_bit == 6:
            x_clipped = torch.clamp(x, -15.5, 15.5)
        elif q_bit == 5:
            x_clipped = torch.clamp(x, -7.5, 7.5)
        elif q_bit == -5:
            x_clipped = torch.clamp(x, -15, 15)
        elif q_bit == 4:
            x_clipped = torch.clamp(x, -7, 7)
        elif q_bit == 3:
            x_clipped = torch.clamp(x, -6, 6)
        
        return x_clipped + (q_value - x_clipped).detach()

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

    def forward(
            self,
            xa: Optional[torch.Tensor | list[torch.Tensor]],
            target_iter: Optional[int | list[int]] = None,
            fixed_iter: Optional[int | list[int]] = None,
            fixed_iter_weight: Optional[torch.Tensor | list[torch.Tensor]] = None,
        ):
        is_input_iterable = target_iter == None and isinstance(target_iter, list)
        if is_input_iterable:
            assert self.iter_node_counts == len(xa)
            assert isinstance(xa[0], torch.Tensor)

        iteration: list[int] = None
        if isinstance(target_iter, int):
            iteration = [target_iter]
        elif isinstance(target_iter, list):
            iteration = target_iter
        else:
            iteration = list(range(self.iter_node_counts))

        fixed_iteration_index = 0
        fixed_iteration: list[int] = []
        if isinstance(fixed_iter, int):
            fixed_iteration = [fixed_iter]
        elif isinstance(fixed_iter, list):
            fixed_iteration = fixed_iter
        if len(fixed_iteration) > 0:
            assert len(fixed_iteration) == len(fixed_iter_weight)
        
        xa_input: torch.Tensor = None
        if not is_input_iterable:
            xa_input = xa.transpose(1, 2)  # [batch, Z, N]

        for curr_iter in iteration:
            if is_input_iterable:
                xa_input = xa[curr_iter].transpose(1, 2)  # [batch, Z, N]

            vn_weight = self.fetch_param(ParamType.Weight, NodeType.VN, curr_iter)

            if self.node_weight_sharing_config.get(NodeType.VN) == 2 or \
               self.node_weight_sharing_config.get(NodeType.VN) == 3:
                xa_input = torch.mul(xa_input, vn_weight)
            elif self.node_weight_sharing_config.get(NodeType.VN) == 4:
                if curr_iter not in fixed_iteration:
                    xa_input = torch.mul(xa_input, vn_weight)
                else:
                    xa_input = torch.mul(xa_input, fixed_iter_weight[fixed_iteration_index])

            if self.decoding_type == DecoderType.QMS:
                xa_input = self._quantize_message(xa_input, self.decoder_qms_qbit)
            
            if self.node_weight_sharing_config.get(NodeType.UCN) > 0:
                if curr_iter == 0:
                    VN_APP = xa_input   # [B, Z, N]
                else:
                    VN_APP = self.outputs[curr_iter - 1].reshape(self.batch_size, self.N, self.Z)  # [B, N, Z]
                    # WARN: T :
                    VN_APP = VN_APP.transpose(1, 2)  # [B, Z, N]
                
                VN_APP = -VN_APP
                VN_APP_sign = torch.add((VN_APP > 0).float(), -(VN_APP <= 0).float())  # [B, Z, N]
                VN_APP_sign_edge = torch.matmul(VN_APP_sign, self.W_skipconn2even)  # [B, Z, E]
                VN_APP_sign_edge = VN_APP_sign_edge.transpose(1, 2)  # [B, E, Z]
                VN_APP_sign_edge = VN_APP_sign_edge.reshape(self.batch_size, self.sum_edge * self.Z)  # [B, E * Z]
                CN_in_sign = torch.matmul(VN_APP_sign_edge, self.Lift_Matrix1.t())  # [B, E * Z]
                # WARN: T :
                CN_in_sign = CN_in_sign.reshape(self.batch_size, self.sum_edge, self.Z)  # [B, E, Z]
                CN_in_sign = CN_in_sign.transpose(1, 2)  # [B, Z, E]
                CN_in_sign_tile = torch.tile(CN_in_sign, (1, 1, self.sum_edge))  # [B, Z, E, E]
                CN_in_sign_tile = torch.mul(CN_in_sign_tile, self.W_even2odd_with_self.t().reshape(-1))  # [B, Z, E, E]
                CN_in_sign_tile = torch.reshape(CN_in_sign_tile, (self.batch_size, self.Z, self.sum_edge, self.sum_edge))
                CN_in_sign_tile = torch.add(CN_in_sign_tile, torch.mul(1.0, torch.add(-torch.abs(CN_in_sign_tile > 0).float(), 1.0)))
                CN_sign_edge = torch.prod(CN_in_sign_tile, dim=3)
                UCN_idx_edge = (CN_sign_edge < 0).float()  # [B, Z, E]
                UCN_idx_edge = UCN_idx_edge.transpose(1, 2)  # [B, E, Z]
                UCN_idx_edge = UCN_idx_edge.reshape(self.batch_size, self.Z * self.sum_edge)  # [B, E * Z]
                UCN_idx_edge = torch.matmul(UCN_idx_edge, self.Lift_Matrix2)  # [B, E * Z]
                UCN_idx_edge = UCN_idx_edge.reshape(self.batch_size, self.sum_edge, self.Z)  # [B, E, Z]
                UCN_idx_edge = UCN_idx_edge.transpose(1, 2)  # [B, Z, E]
                SCN_idx_edge = torch.add(1.0, -UCN_idx_edge)  # [B, Z, E]
            else:
                UCN_idx_edge = torch.zeros((self.batch_size, self.Z, self.sum_edge), device=self.conn_mat.device, dtype=torch.float32)
                SCN_idx_edge = torch.ones((self.batch_size, self.Z, self.sum_edge), device=self.conn_mat.device, dtype=torch.float32)

            x0 = torch.matmul(xa_input, self.W_skipconn2even)  # [B, Z, E]
            x1 = torch.matmul(self.llr[curr_iter], self.W_odd2even)  # [B, Z, E]
            x2 = torch.add(x0, x1)  # [B, Z, E]

            x2 = x2.transpose(1, 2)  # [B, E, Z]
            x2 = x2.reshape(self.batch_size, self.sum_edge * self.Z)  # [B, E * Z]
            x2 = torch.matmul(x2, self.conn_mat.lifting_matrix_1.t())  # [B, E * Z]
            x2 = x2.reshape(self.batch_size, self.sum_edge, self.Z)  # [B, E, Z]
            x2 = x2.transpose(1, 2)  # [B, Z, E]

            if self.decoding_type == DecoderType.QMS:
                x2 = self._quantize_message(x2, self.decoder_qms_qbit)
            else:
                x2 = torch.clamp(x2, self.allowed_llr_range.start, self.allowed_llr_range.end)
            
            if self.decoding_type == DecoderType.MS or \
                self.decoding_type == DecoderType.QMS:
                x2 = x2 + 0.0001 * (1 - (torch.abs(x2) > 0).float())
            x_tile = torch.tile(x2, (1, 1, self.sum_edge))  # [B, Z, E, E]
            W_input_reshape = self.W_even2odd.t().reshape(-1)  # [E * E]

            x_tile_mul = torch.mul(x_tile, W_input_reshape)  # [B, Z, E, E]
            x2_1 = x_tile_mul.reshape(self.batch_size, self.Z, self.sum_edge, self.sum_edge)  # [B, Z, E, E]

            if self.decoding_type == DecoderType.SP:
                # Sum-product operations
                x2_tanh = torch.tanh(torch.mul(-0.5, x2_1))
                x2_abs = torch.add(x2_tanh, torch.mul(1.0 - torch.abs(x2_tanh > 0).float(), 1.0))
                x3 = torch.prod(x2_abs, dim=3)  # [B, Z, E]

                epsilon = 1e-7
                x3_clipped = torch.clamp(x3, -1 + epsilon, 1 - epsilon)
                x_output_0 = torch.mul(-2.0, torch.atanh(x3_clipped))
            elif self.decoding_type == DecoderType.MS or \
                 self.decoding_type == DecoderType.QMS:
                x2_abs = torch.add(
                    torch.abs(x2_1),
                    torch.mul(10000.0, 1.0 - (x2_1 > 0).float())
                )
                x3 = torch.min(x2_abs, dim=3)[0]  # torch.min returns (values, indices)
                x3 = torch.add(x3, torch.mul(-0.0001, torch.add(-(x3 > 0.0001).float(), 1.0)))
                x2_2 = torch.mul(-1.0, x2_1)
                x4 = torch.add(
                    torch.zeros((self.batch_size, self.Z, self.sum_edge, self.sum_edge), device=self.conn_mat.device, dtype=torch.float32),
                    torch.mul(1.0, torch.add(-(x2_2 > 0).float(), 1.0))
                )
                x4_prod = torch.mul(-1.0, torch.prod(x4, dim=3))  # [B, Z, E]
                x_output_0 = torch.mul(x3, torch.sign(x4_prod))
            
            x_output_0 = x_output_0.transpose(1, 2)  # [B, E, Z]
            x_output_0 = x_output_0.reshape(self.batch_size, self.Z * self.sum_edge)  # [B, E * Z]
            x_output_0 = torch.matmul(x_output_0, self.Lift_Matrix2)  # [B, Z, E]
            x_output_0 = x_output_0.reshape(self.batch_size, self.sum_edge, self.Z)  # [B, E, Z]
            x_output_0 = x_output_0.transpose(1, 2)  # [B, Z, E]

            _cn_sharing = self.node_weight_sharing_config.get(NodeType.CN)
            _ucn_sharing = self.node_weight_sharing_config.get(NodeType.UCN)
            if _cn_sharing == 0:
                x_output_1 = torch.abs(x_output_0)
            elif _cn_sharing == 1:
                if _ucn_sharing == 1:
                    W_per_edge_1 = self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter)
                    W_per_edge_2 = self.fetch_param(ParamType.Weight, NodeType.UCN, curr_iter)
                    x_output_11 = torch.mul(torch.abs(x_output_0), W_per_edge_1)
                    x_output_12 = torch.mul(torch.abs(x_output_0), W_per_edge_2)
                    x_output_1 = torch.add(torch.mul(x_output_11, SCN_idx_edge), torch.mul(x_output_12, UCN_idx_edge))
                else:
                    W_per_edge = self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter)
                    x_output_1 = torch.mul(torch.abs(x_output_0), W_per_edge)
            elif _cn_sharing == 2:
                if _ucn_sharing == 2:
                    W_per_edge_1 = torch.matmul(
                        self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter).view(1, -1),
                        self.W_skipconn2odd
                    )
                    W_per_edge_2 = torch.matmul(
                        self.fetch_param(ParamType.Weight, NodeType.UCN, curr_iter).view(1, -1),
                        self.W_skipconn2odd
                    )
                    x_output_11 = torch.mul(torch.abs(x_output_0))
                    x_output_12 = torch.mul(torch.abs(x_output_0))
                    x_output_1 = torch.add(
                        torch.mul(x_output_11, -UCN_idx_edge + 1.0),
                        torch.mul(x_output_12, UCN_idx_edge)
                    )
                else:
                    W_per_edge = torch.matmul(
                        self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter).view(1, -1),
                        self.W_skipconn2odd
                    )
                    x_output_1 = torch.mul(torch.abs(x_output_0), W_per_edge)  # [B, Z, E]
            elif _cn_sharing == 3:
                if _ucn_sharing == 3:
                    W_per_edge_1 = torch.matmul(
                        torch.tile(
                            self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter),
                            [self.M]
                        ).view(1, self.M),
                        self.W_skipconn2odd
                    )
                    W_per_edge_2 = torch.matmul(
                        torch.tile(
                            self.fetch_param(ParamType.Weight, NodeType.UCN, curr_iter),
                            [self.M]
                        ).view(1, self.M),
                        self.W_skipconn2odd
                    )
                    x_output_11 = torch.mul(torch.mul(torch.abs(x_output_0)), W_per_edge_1)
                    x_output_12 = torch.mul(torch.mul(torch.abs(x_output_0)), W_per_edge_2)
                    x_output_1 = torch.add(
                        torch.mul(x_output_11, SCN_idx_edge),
                        torch.mul(x_output_12, UCN_idx_edge)
                    )
                else:
                    W_per_edge = torch.matmul(
                        torch.tile(
                            self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter),
                            [self.M]
                        ).view(1, self.M),
                        self.W_skipconn2odd
                    )
                    x_output_1 = torch.mul(torch.abs(x_output_0), W_per_edge)  # [B, Z, E]
            elif _cn_sharing == 4:
                if curr_iter not in fixed_iteration:
                    W_per_edge = self.fetch_param(ParamType.Weight, NodeType.CN, curr_iter)
                else:
                    W_per_edge = fixed_iter_weight[fixed_iteration_index]
                x_output_1 = torch.mul(torch.abs(x_output_0), W_per_edge)  # [B, Z, E]

            x_output_2 = torch.mul(x_output_1, (x_output_0 >= 0).float())  # [B, Z, E]

            if self.decoding_type == DecoderType.QMS:
                x_output_2 = self._quantize_message(x_output_2, self.decoder_qms_qbit)
            else:
                x_output_2 = torch.clamp(x_output_2, self.allowed_llr_range.start, self.allowed_llr_range.end)
            
            self.llr[curr_iter + 1] = torch.mul(x_output_2, torch.sign(x_output_0))  # [B, Z, E]
            y_output_2 = torch.matmul(self.llr[curr_iter + 1], self.W_output)  # [B, Z, N]
            y_output_3 = y_output_2.transpose(1, 2)  # [B, N, Z]

            # decision
            if self.decoding_type == DecoderType.QMS:
                xa = self._quantize_message(xa, self.decoder_qms_qbit)
            
            y_output_4 = torch.add(xa_input, y_output_3)  # [B, N, Z]
            y_output_4 = torch.clamp(y_output_4, self.allowed_llr_range.start, self.allowed_llr_range.end)

            self.outputs[curr_iter] = torch.reshape(
                y_output_4,
                (self.batch_size, self.N * self.Z)
            )  # [B, N * Z]
                    

            # Postprocess for next iteration
            fixed_iteration_index += 1

        return self.outputs
