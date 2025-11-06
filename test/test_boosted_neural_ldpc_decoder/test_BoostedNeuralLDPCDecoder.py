import unittest
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from math import floor

from boosted_neural_ldpc_decoder import ConnectingMatrixTorch, ConnectingMatrix, AWGNPassedDatagen
from boosted_neural_ldpc_decoder.BoostedNeuralLDPCDecoder import BoostedNeuralLDPCDecoder
from boosted_neural_ldpc_decoder.struct.Clipping import Clipping
from boosted_neural_ldpc_decoder.struct.DecoderType import DecoderType
from boosted_neural_ldpc_decoder.LDPCDecoderLoss import LDPCDecoderLoss
from boosted_neural_ldpc_decoder.struct.LearningRate import LearningRate
from boosted_neural_ldpc_decoder.struct.NodeWeightSharingConfig import NodeWeightSharingConfig
from boosted_neural_ldpc_decoder.struct.Puncture import Puncture
from boosted_neural_ldpc_decoder.struct.Shortening import Shortening


class test_BoostedNeuralLDPCDecoder(unittest.TestCase):
    def test_boosted_neural_ldpc_decoder(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")
        device = torch.device(device)

        # Graph
        # basegraph = np.loadtxt("resources/wman_N0576_R34_z24.txt", int, delimiter="\t")
        # Z = 24
        basegraph = np.loadtxt("resources/basegraph2_set0.txt", int, delimiter="\t")
        genmatrix = np.loadtxt("resources/gen_matrix_bg2_z16.txt", int, delimiter=",")
        Z = 16

        conn = ConnectingMatrixTorch(
            ConnectingMatrix(
                Z=Z,
                basegraph=basegraph,
                dtype_w_odd2even=np.float32,
                dtype_w_skipconn2even=np.float32,
                dtype_w_even2odd=np.float32,
                dtype_w_output=np.float32,
                dtype_lifting_matrix=np.float32
            ),
            device=device,
            dtype_w_odd2even=torch.float32,
            dtype_w_skipconn2even=torch.float32,
            dtype_w_even2odd=torch.float32,
            dtype_w_output=torch.float32,
            dtype_lifting_matrix=torch.float32
        )

        node_weight_sharing_config = NodeWeightSharingConfig(
            cn_weight_sharing=3,
            ucn_weight_sharing=0,
            vn_weight_sharing=3
        )

        decoding_type = DecoderType.QMS
        decoder_qms_qbit = 5

        puncturing = Puncture(
            start=0,
            end=0
        )
        shortening = Shortening(
            start=0,
            end=0
        )

        allowed_weight_range = Clipping(start=0, end=2)
        allowed_bias_range = Clipping(start=0, end=2)
        # reserved: not used actually

        allowed_llr_range = Clipping(abs=20.0)

        iter_node_counts = 20

        # AWGN
        snr_matrix = np.array([2, 2.5, 3.0, 3.5, 4.0])

        fixed_iterative_nodes: list[int] = []
        """
        List of Fixed iteration Nodes
        in this model, iterations are identified by their index (0-based).
        if you want to fix the weights of certain iterations, add their indices to this list.
        i.e. `fixed_iters = [i for i in range(3, iter_node_counts, 3)]`
        """
        fixed_iterative_nodes_init_weight = 0

        # Random generator
        awgn_noise_seed: int = 2042
        wordgen_random_seed: int = 1074

        # Model
        batch_size = word_length = 50

        train_word_length = 100

        # Training
        learning_rate = LearningRate(
            initial_lr=.001,
            decay_rate=0,
            decay_steps=0,
        )
        train_is_y_all_zero = False
        train_total_epochs = 20000

        model = BoostedNeuralLDPCDecoder(
            iter_node_counts=iter_node_counts,
            batch_size=batch_size,
            connecting_matrix=conn,
            node_weight_sharing_config=node_weight_sharing_config,
            decoding_type=decoding_type,
            decoder_qms_qbit=decoder_qms_qbit,
            fixed_iterative_nodes=fixed_iterative_nodes,
            fixed_iterative_nodes_init_weight=fixed_iterative_nodes_init_weight,
            allowed_weight_range=allowed_weight_range,
            allowed_bias_range=allowed_bias_range,
            allowed_llr_range=allowed_llr_range,
            dtype_cn_weight=torch.float32,
            dtype_ucn_weight=torch.float32,
            dtype_vn_weight=torch.float32,
            init_cn_weight=1,
            init_ucn_weight=1,
            init_vn_weight=1,
            dtype_cn_bias=torch.float32,
            dtype_ucn_bias=torch.float32,
            dtype_vn_bias=torch.float32,
            init_cn_bias=1,
            init_ucn_bias=1,
            init_vn_bias=1,
        ).to(device)

        criterion = LDPCDecoderLoss()
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate())

        N, M = conn.N, conn.M
        datagen = AWGNPassedDatagen(
            N=N,
            M=M,
            snr_db=snr_matrix,
            awgn_noise_seed=awgn_noise_seed,
            wordgen_random_seed=wordgen_random_seed,
            gen_matrix=genmatrix,
            # decoding_type=decoding_type,
            # decoding_qms_q_bit=decoder_qms_qbit,
            # puncture=puncturing,
            # shortening=shortening,
            # allowed_llr_range=Clipping(abs=20.0),
        )

        for epoch in range(train_total_epochs):
            x, y = [], []
            batch_size_per_word_length = floor(train_word_length / batch_size)
            for iteration in range(iter_node_counts):
                for _i in range(batch_size_per_word_length):
                    if not x or not y:
                        x, y = datagen(
                            word_length=batch_size,
                            Z=Z,
                            is_y_all_zero=train_is_y_all_zero,
                        )
                    x_i, y_i = x.pop(0), y.pop(0)

                    x_i = np.reshape(x_i, [batch_size, N, Z])
                    y_i = np.reshape(y_i, [batch_size, N * Z])

                    x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
                    y_i = torch.tensor(y_i, dtype=torch.float32, device=device)

                    model.train()
                    outputs = model(x_i)

                    loss = criterion(outputs[iteration], y_i)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epoch % 100 == 0:
                    print(f"epoch {epoch}/{train_total_epochs}, iter {iteration}, loss {loss.item()}")
            if epoch % 10 == 0:
                print(f"Cycle {epoch} completed at {datetime.now()}")


if __name__ == '__main__':
    unittest.main()
