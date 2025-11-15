from sys import stdout
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from math import floor

from boosted_neural_ldpc_decoder import ConnectingMatrixTorch, ConnectingMatrix, AWGNPassedDatagen, Functions
from boosted_neural_ldpc_decoder.BoostedNeuralLDPCDecoder import BoostedNeuralLDPCDecoder
from boosted_neural_ldpc_decoder.struct.Clipping import Clipping
from boosted_neural_ldpc_decoder.struct.DecoderType import DecoderType
from boosted_neural_ldpc_decoder.LDPCDecoderLoss import LDPCDecoderLoss
from boosted_neural_ldpc_decoder.struct.LearningRate import LearningRate
from boosted_neural_ldpc_decoder.struct.LossType import LossType
from boosted_neural_ldpc_decoder.struct.NodeWeightSharingConfig import NodeWeightSharingConfig
from boosted_neural_ldpc_decoder.struct.Puncture import Puncture
from boosted_neural_ldpc_decoder.struct.Shortening import Shortening
from checkpoint_utils import CheckPointUtil
from checkpoint_utils import MetricsLogger

def print_train_progress(
        current_batch,
        total_batches,
        current_epoch,
        total_epochs,
        loss=None,
        start_time=None,
        bar_length=40
    ):
    percent = current_epoch / total_epochs
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    progress_str = f'[{timestamp}] Epoch {current_epoch}/{total_epochs} [{bar}] {current_batch}/{total_batches}'
    
    if loss is not None:
        progress_str += f' Loss: {loss:.6f}'
    
    # Calculate ETA
    if start_time is not None and current_batch > 0:
        elapsed = datetime.now().timestamp() - start_time
        batches_done = (current_epoch - 1) * total_batches + current_batch
        total_batches_needed = total_epochs * total_batches
        
        if batches_done > 0:
            avg_time_per_batch = elapsed / batches_done
            remaining_batches = total_batches_needed - batches_done
            eta_seconds = remaining_batches * avg_time_per_batch
            
            # Format ETA
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            seconds = int(eta_seconds % 60)
            
            if hours > 0:
                eta_str = f"{hours}h {minutes}m"
            elif minutes > 0:
                eta_str = f"{minutes}m {seconds}s"
            else:
                eta_str = f"{seconds}s"
            
            progress_str += f' ETA: {eta_str}'
    
    stdout.write('\r' + progress_str)
    stdout.flush()

    if current_batch == total_batches:
        stdout.write('\n')

def train_boosted_neural_ldpc_decoder(
        param_train_total_epochs: int = 500,
        param_train_is_y_all_zero: bool = False,
):
    # Configure torch device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    device = torch.device(device)

    # Load graph and generator matrix
    """
    A generator matrix is needed when generating a dataset where y is not all-zero.
    If you don't need to use a non-all-zero dataset, genmatrix is not required.
    But using a generator matrix is required to validate whether the model is going to overfit or not.
    In my case, there is some wrong forward pass logic, and the model just overfitted
    to translate all input into zero.
    """
    basegraph = np.loadtxt("resources/basegraph2_set0.txt", int, delimiter="\t")
    genmatrix = np.loadtxt("resources/gen_matrix_bg2_z16.txt", int, delimiter=",")
    Z = 16

    # basegraph = np.loadtxt("resources/wman_N0576_R34_z24.txt", int, delimiter="\t")
    # genmatrix = None
    # Z = 24

    # Initialize connecting matrix
    """
    The connecting matrix would be generated when constructing the ConnectingMatrix object.
    Because ConnectingMatrix depends on numpy, ConnectingMatrixTorch supports using numpy.
    """
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
    train_word_input_length = 10000
    validate_word_input_length = 1000

    # Training
    loss_type = LossType.BCE
    etha_init = 1.0  # Exponential weighting: 0=last iter only, 1.0=equal weight, >1=favor early iters
    learning_rate = LearningRate(
        initial_lr=0.001,
        decay_rate=0,
        decay_steps=0,
    )
    train_is_y_all_zero = param_train_is_y_all_zero
    train_total_epochs = param_train_total_epochs

    # Early stopping
    patience = 10
    min_delta = 1e-5
    best_loss = float('inf')
    patience_counter = 0

    # Validating
    # Setting checkpoint_step and log_metrics_step into multiple of validate_epoch step is recommended.
    validate_epoch_step = 5
    checkpoint_step = validate_epoch_step
    log_metrics_step = 5
    train_progress_inform_step = 5

    # Training iteration parameters
    training_iter_start = fixed_iter
    training_iter_end = fixed_iter + iter_step

    # Initialize data generator
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

    # Initialize Model
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

    # Initialize loss function with multi-iteration weighting
    criterion = LDPCDecoderLoss(
        loss_type=loss_type,
        etha=etha_init,
    )

    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate())

    checkpoint_util = CheckPointUtil(checkpoint_dir="checkpoints")
    metrics_logger = MetricsLogger(log_dir="checkpoints")

    # Initializing Done
    
    # Training
    training_batch_size = floor(train_word_input_length / batch_size)

    # Metrics: define variables that used when validating previously
    avg_valid_loss = 0
    last_iter_ber = 0
    last_iter_fer = 0
    
    # Training Loop
    training_start_time = datetime.now().timestamp()
    
    avg_epoch_loss = 0.0
    for epoch in range(train_total_epochs + 1):
        epoch_loss = 0.0

        current_lr = learning_rate.lr

        if epoch > 0:
            current_lr = learning_rate()
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            for batch_idx in range(training_batch_size):
                x_i, y_i = datagen(
                    gentype="mix_snr",
                    word_length=batch_size,
                    Z=Z,
                    is_y_all_zero=train_is_y_all_zero,
                    decoding_type=decoding_type,
                    decoder_qms_qbit=decoder_qms_qbit,
                )
                
                x_i = np.reshape(x_i, [batch_size, N, Z])
                x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
                y_i = torch.tensor(y_i, dtype=torch.float32, device=device)
                
                model.train()
                optimizer.zero_grad()
                
                # NOTE: Single forward pass through ALL iterations with this batch's data
                all_outputs = model(
                    x_i,  # Same x_i for all iterations
                    target_iter=list(range(training_iter_start, training_iter_end))
                )
                
                # Compute loss on all iteration outputs
                total_loss = criterion(
                    all_outputs,
                    y_i,
                    coeff_param=list(range(len(all_outputs)))
                )
                
                # Single backward pass propagates gradients through ALL iterations
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model._apply_constraints()
                
                epoch_loss += total_loss.item()

                if batch_idx % train_progress_inform_step == 0:
                    print_train_progress(
                        current_batch=batch_idx + 1,
                        total_batches=training_batch_size,
                        current_epoch=epoch,
                        total_epochs=train_total_epochs,
                        loss=total_loss.item(),
                        start_time=training_start_time
                    )
            
            if training_batch_size > 0:
                print_train_progress(
                    current_batch=training_batch_size,
                    total_batches=training_batch_size,
                    current_epoch=epoch,
                    total_epochs=train_total_epochs,
                    loss=total_loss.item(),
                    start_time=training_start_time
                )

            avg_epoch_loss = epoch_loss / training_batch_size if training_batch_size > 0 else 0.0
            stdout.write('\n')
            stdout.flush()
            print(f"{'=' * 60}")
            print(f"Epoch {epoch}/{train_total_epochs} Complete")
            print(f"Training iter range: [{training_iter_start}, {training_iter_end})")
            print(f"Average training loss: {avg_epoch_loss:.6f}")
            print(f"{'='* 60}\n")
                
        # Validation
        if epoch % validate_epoch_step == 0:
            model.eval()
            
            with torch.no_grad():
                valid_batch_size = floor(validate_word_input_length / batch_size)
                valid_loss = 0.0
                
                # BER/FER tracking
                total_bits = 0
                total_bit_errors = 0
                total_frames = 0
                total_frame_errors = 0
                last_iter_total_bits = 0
                last_iter_total_bit_errors = 0
                last_iter_total_frames = 0
                last_iter_total_frame_errors = 0
                
                for batch_idx in range(valid_batch_size):
                    x_i, y_i = datagen(
                        gentype="mix_snr",
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
                    print(outputs)
                    loss = criterion(outputs, y_i)
                    valid_loss += loss.item()
                    
                    (bit_errors, bits), (frame_errors, frames) = Functions.evaluate_ber_fer(y_i, outputs)
                    
                    if batch_idx == 0:
                        per_iter_ber = [be / bits for be in bit_errors]
                        per_iter_fer = [fe / frames for fe in frame_errors]
                        print(f">>> Per-Iteration Performance (First Validation Batch):")
                        best_ber_idx = per_iter_ber.index(min(per_iter_ber))
                        for i, (b, f) in enumerate(zip(per_iter_ber, per_iter_fer)):
                            marker = " ← BEST BER" if i == best_ber_idx else ""
                            print(f"    Iter {i:2d}: BER={b:.6e}, FER={f:.4f}{marker}")
                        print()

                    total_bit_errors += sum(bit_errors)
                    total_bits += bits * len(bit_errors)
                    total_frame_errors += sum(frame_errors)
                    total_frames += frames * len(frame_errors)

                    last_iter_total_bit_errors += bit_errors[-1]
                    last_iter_total_bits += bits
                    last_iter_total_frame_errors += frame_errors[-1]
                    last_iter_total_frames += frames
                
                avg_valid_loss = valid_loss / valid_batch_size
                ber = total_bit_errors / total_bits if total_bits > 0 else 0
                fer = total_frame_errors / total_frames if total_frames > 0 else 0
                last_iter_ber = last_iter_total_bit_errors / last_iter_total_bits if last_iter_total_bits > 0 else 0
                last_iter_fer = last_iter_total_frame_errors / last_iter_total_frames if last_iter_total_frames > 0 else 0

                stdout.write('\n')
                stdout.flush()
                print(f">>> Validation Results (Epoch {epoch})")
                print(f">>> Learning rate: {current_lr:.6e}")
                print(f">>> Validation loss: {avg_valid_loss:.6f}")
                print(f">>> BER(entire iter): {ber:.6e} ({total_bit_errors:.0f}/{total_bits})")
                print(f">>> FER(entire iter): {fer:.6f} ({total_frame_errors:.0f}/{total_frames})")
                print(f">>> BER(last iter): {last_iter_ber:.6e} ({last_iter_total_bit_errors:.0f}/{last_iter_total_bits})")
                print(f">>> FER(last iter): {last_iter_fer:.6f} ({last_iter_total_frame_errors:.0f}/{last_iter_total_frames})")
                print()

                if avg_valid_loss < best_loss - min_delta:
                    best_loss = avg_valid_loss
                    patience_counter = 0
                    print(f">>> New best loss: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f">>> No improvement ({patience_counter}/{patience})")
                    
                    if patience_counter >= patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch}")
                        print(f"Best loss: {best_loss:.6f}")
                        print(f"{'='*60}\n")
                        break

        # During training/validation:
        # metrics are defined before the loop entry
        # but if validation is not executed in these loop,
        # metrics dictionary refers legacy metrics
        metrics = {
            'loss': avg_epoch_loss,
            'ber_last_iter': last_iter_ber,
            'fer_last_iter': last_iter_fer,
        }
        if epoch % validate_epoch_step == 0:
            metrics['loss'] = avg_valid_loss
            
        checkpoint_dumping_cfg = { "batch_size": batch_size, "lr": current_lr }
        checkpoint_filename = "NA"

        # Check point
        if epoch % checkpoint_step == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
            checkpoint_util.save(
                filepath=checkpoint_filename,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=checkpoint_dumping_cfg
            )

            checkpoint_util.save_weights(
                filepath=f"weights_epoch_{epoch:04d}",
                model=model,
                as_txt=True
            )
        
        # Log metrics
        if epoch % log_metrics_step == 0:
            metrics_logger.log(
                epoch=epoch,
                metrics=metrics,
                checkpoint_filename=checkpoint_filename,
                config=checkpoint_dumping_cfg
            )

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Boosted Neural LDPC Decoder")
    parser.add_argument('--epochs', type=int, default=500, help='Total number of training epochs')
    parser.add_argument('--y_all_zero', action='store_true', help='Use all-zero codewords for training')
    args = parser.parse_args()
    train_boosted_neural_ldpc_decoder(
        param_train_total_epochs=args.epochs,
        param_train_is_y_all_zero=args.y_all_zero
    )
