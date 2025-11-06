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

        puncturing = Puncture(start=0, end=0)
        shortening = Shortening(start=0, end=0)
        allowed_weight_range = Clipping(start=0, end=2)
        allowed_bias_range = Clipping(start=0, end=2)
        allowed_llr_range = Clipping(abs=20.0)

        iter_node_counts = 20
        fixed_iter = 0
        fixed_init = 0  # Delta_2
        iter_step = 20  # Delta_1

        # AWGN
        snr_matrix = np.array([2, 2.5, 3.0, 3.5, 4.0])

        # Random generator
        awgn_noise_seed: int = 2042
        wordgen_random_seed: int = 1074

        # Model
        batch_size = 20
        train_word_length = 10000

        # Training
        loss_type = 0  # 0: BCE (works for zero and non-zero), 1: Soft BER (zero only), 2: FER (zero only)
        etha_start = 1.0  # Exponential weighting: 0=last iter only, 1.0=equal weight, >1=favor early iters
        learning_rate = LearningRate(
            initial_lr=.001,
            decay_rate=0,
            decay_steps=0,
        )
        train_is_y_all_zero = False
        train_total_epochs = 200

        # Validating
        validate_epoch_step = 1

        # Training iteration parameters
        training_iter_start = fixed_iter
        training_iter_end = fixed_iter + iter_step

        N, M = conn.N, conn.M
        datagen = AWGNPassedDatagen(
            N=N,
            M=M,
            snr_db=snr_matrix,
            awgn_noise_seed=awgn_noise_seed,
            wordgen_random_seed=wordgen_random_seed,
            gen_matrix=genmatrix,
            puncturing=puncturing,
            shortening=shortening,
            allowed_llr_range=allowed_llr_range,
        )

        fixed_iterative_nodes: list[int] = []
        model = BoostedNeuralLDPCDecoder(
            iter_node_counts=iter_node_counts,
            batch_size=batch_size,
            connecting_matrix=conn,
            node_weight_sharing_config=node_weight_sharing_config,
            decoding_type=decoding_type,
            decoder_qms_qbit=decoder_qms_qbit,
            fixed_iterative_nodes=fixed_iterative_nodes,
            fixed_iterative_nodes_init_weight=0,
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

        # Create loss function with multi-iteration weighting
        criterion = LDPCDecoderLoss(
            loss_type=loss_type,
            etha=etha_start,
            training_iter_start=training_iter_start,
            training_iter_end=training_iter_end,
            fixed_init=fixed_init,
            fixed_iter=fixed_iter,
        )
        
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate())

        # Training loop
        training_batch_num = floor(train_word_length / batch_size)
        
        for epoch in range(train_total_epochs + 1):
            epoch_loss = 0.0
            
            if epoch > 0:  # Skip training on epoch 0 (just validation)
                for batch_idx in range(training_batch_num):
                    # Generate MIXED SNR batch (matches TensorFlow)
                    x_i, y_i = datagen._gendata_mixed(
                        word_length=batch_size,
                        Z=Z,
                        is_y_all_zero=train_is_y_all_zero,
                        decoding_type=decoding_type,
                        decoder_qms_qbit=decoder_qms_qbit,
                    )
                    
                    # Reshape for model input
                    x_i = np.reshape(x_i, [batch_size, N, Z])
                    
                    # Convert to tensors
                    x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
                    y_i = torch.tensor(y_i, dtype=torch.float32, device=device)
                    
                    # Forward pass
                    model.train()
                    outputs = model(x_i)
                    
                    # Compute loss
                    loss = criterion(outputs, y_i)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Print progress every 50 batches
                    if batch_idx % 50 == 0:
                        print(f"Epoch {epoch}/{train_total_epochs}, "
                              f"Batch {batch_idx}/{training_batch_num}, "
                              f"Loss: {loss.item():.6f}")
                
                avg_epoch_loss = epoch_loss / training_batch_num
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch}/{train_total_epochs} Complete")
                print(f"Training iter range: [{training_iter_start}, {training_iter_end})")
                print(f"Average training loss: {avg_epoch_loss:.6f}")
                print(f"{'='* 60}\n")
            
            # Validation
            if epoch % validate_epoch_step == 0:
                model.eval()
                with torch.no_grad():
                    valid_num = 1000
                    valid_batch_num = floor(valid_num / batch_size)
                    valid_loss = 0.0
                    
                    # BER/FER tracking
                    total_bits = 0
                    total_bit_errors = 0
                    total_frames = 0
                    total_frame_errors = 0
                    
                    for batch_idx in range(valid_batch_num):
                        # Use mixed SNR for validation (consistent with training)
                        x_i, y_i = datagen._gendata_mixed(
                            word_length=batch_size,
                            Z=Z,
                            is_y_all_zero=train_is_y_all_zero,
                            decoding_type=decoding_type,
                            decoder_qms_qbit=decoder_qms_qbit,
                        )
                        
                        x_i = np.reshape(x_i, [batch_size, N, Z])
                        x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
                        y_i = torch.tensor(y_i, dtype=torch.float32, device=device)
                        
                        outputs = model(x_i)
                        loss = criterion(outputs, y_i)
                        valid_loss += loss.item()
                        
                        # Calculate BER/FER on final iteration output
                        final_output = outputs[-1]  # Last iteration output [batch, N*Z]
                        
                        # Hard decision: sign of LLR
                        # Positive LLR → decoded bit = 0
                        # Negative LLR → decoded bit = 1
                        # So: decoded_bits = (LLR < 0).float()
                        decoded_bits = (final_output < 0).float()
                        
                        # Compare with target
                        bit_errors = (decoded_bits != y_i).float()
                        
                        # BER: bit error rate
                        batch_bit_errors = bit_errors.sum().item()
                        batch_bits = y_i.numel()
                        total_bit_errors += batch_bit_errors
                        total_bits += batch_bits
                        
                        # FER: frame error rate (any bit error in a frame)
                        frame_errors = (bit_errors.sum(dim=1) > 0).float()
                        batch_frame_errors = frame_errors.sum().item()
                        batch_frames = y_i.shape[0]
                        total_frame_errors += batch_frame_errors
                        total_frames += batch_frames
                    
                    avg_valid_loss = valid_loss / valid_batch_num
                    ber = total_bit_errors / total_bits if total_bits > 0 else 0
                    fer = total_frame_errors / total_frames if total_frames > 0 else 0
                    
                    print(f">>> Validation Results (Epoch {epoch})")
                    print(f">>> Validation loss: {avg_valid_loss:.6f}")
                    print(f">>> BER: {ber:.6e} ({total_bit_errors:.0f}/{total_bits})")
                    print(f">>> FER: {fer:.6f} ({total_frame_errors:.0f}/{total_frames})\n")

if __name__ == '__main__':
    unittest.main()
