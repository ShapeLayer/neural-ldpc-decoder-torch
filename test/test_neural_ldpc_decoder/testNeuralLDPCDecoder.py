import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import neural_ldpc_decoder.AWGNPassedDatagen as AWGNPassedDatagen
import neural_ldpc_decoder.ConnectingMatrix as ConnectingMatrix
import neural_ldpc_decoder.ConnectingMatrixTorch as ConnectingMatrixTorch
import neural_ldpc_decoder.NeuralLDPCDecoder as NeuralLDPCDecoder


class MyTestCase(unittest.TestCase):
    def test_neural_ldpc_decoder(self):
        # Computing
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")
        device = torch.device(device)

        # Graph
        basegraph = np.loadtxt("resources/basegraph2_set0.txt", int, delimiter="\t")
        genmatrix = np.loadtxt("resources/gen_matrix_bg2_z16.txt", int, delimiter=",")

        Z = 16
        N = 52
        M = 42
        K = N - M

        iter_node_counts = 25

        # AWGN
        code_rate = 1.0 * (N - M) / (N - 2)
        snr = np.array([9.0, 6.05, 4.1, 2.95, 2.25, 1.8, 1.55, 1.3, 1.15, 1.05, 0.94, 0.85, 0.83, 0.81, 0.8, 0.8, 0.8, 0.75, 0.75, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        snr_lin = 10.0 ** (snr / 10.0)
        snr_sigma = np.sqrt(1.0 / (2.0 * snr_lin * code_rate))

        # Random generator
        awgn_noise_seed: int = 2042
        wordgen_random_seed: int = 1074

        # Model
        batch_size = word_length = 50

        # Training
        learning_rate = .005
        train_is_y_all_zero = True
        train_total_epochs = 100

        conn = ConnectingMatrixTorch(
            ConnectingMatrix(
                N=N,
                M=M,
                Z=Z,
                basegraph=basegraph.copy(),
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
        model = NeuralLDPCDecoder(
            iter_node_counts=iter_node_counts,
            batch_size=batch_size,
            connecting_matrix=conn
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizers = [optim.Adam([model.weights_var[i], model.biases_var[i]], lr=learning_rate) for i in range(iter_node_counts)]

        datagen = AWGNPassedDatagen(
            snr_db=snr.tolist(),
            awgn_noise_seed=awgn_noise_seed,
            wordgen_random_seed=wordgen_random_seed,
            x_dtype=np.float32,
            y_dtype=np.int64,
            gen_matrix=genmatrix
        )

        for epoch in range(train_total_epochs):
            for iteration in range(iter_node_counts):
                snr_set = np.array([snr_sigma[iteration]])
                xa, ya = datagen(
                    word_length=word_length,
                    N=N,
                    K=K,
                    Z=Z,
                    scaling_factor=snr_set.tolist(),
                    is_y_all_zero=train_is_y_all_zero,
                )

                xa = np.reshape(xa, [batch_size, N, Z])
                ya = np.reshape(ya, [batch_size, N * Z])

                xa = torch.tensor(xa, dtype=torch.float32, device=device)
                ya = torch.tensor(ya, dtype=torch.float32, device=device)

                model.train()
                outputs = model(xa)

                loss = criterion(outputs[iteration], ya)
                optimizers[iteration].zero_grad()
                loss.backward()
                optimizers[iteration].step()

                if epoch % 10 == 0:
                    print(f"epoch {epoch}/{train_total_epochs}, iter {iteration}, loss {loss.item()}")
            print(f"Cycle {epoch} completed at {datetime.now()}")

if __name__ == '__main__':
    unittest.main()
