from typing import Any

import numpy as np
from numpy import ndarray, dtype
from numpy.random import RandomState

from boosted_neural_ldpc_decoder.Functions import Functions
from boosted_neural_ldpc_decoder.struct.Clipping import Clipping
from boosted_neural_ldpc_decoder.struct.DecoderType import DecoderType
from boosted_neural_ldpc_decoder.struct.Puncture import Puncture
from boosted_neural_ldpc_decoder.struct.Shortening import Shortening


class AWGNPassedDatagen:
    """
    Create a dataset for training or testing a neural network decoder.

    Transferring this function into PyTorch Dataset is considered, but cancelled.
    PyTorch Dataset is effective when the dataset is already generated and stored,
    cannot be generated in realtime. In many cases, the dataset object requires
    a lot of memory space. But this dataset can be generated in real time, and there
    is no need to store data in the memory that already be used.
    """

    def __init__(
            self,
            N: int,
            M: int,
            snr_db: np.ndarray,

            awgn_noise_seed: int = 2042,
            wordgen_random_seed: int = 1074,

            x_dtype=np.float32,
            y_dtype=np.int64,

            gen_matrix: np.ndarray = None,

            puncturing: Puncture = Puncture(0, 0),
            shortening: Shortening = Shortening(0, 0),
            allowed_llr_range: Clipping = Clipping(abs=20.0),
    ):
        self.N = N
        self.M = M
        self.K = N - M
        self.snr_db = snr_db
        self.code_rate = 1.0 * self.K / (N - len(puncturing) - len(shortening))
        self.snr_lin = 10.0 ** (self.snr_db / 10.0)
        self.snr_sigma = np.sqrt(1.0 / (2.0 * self.snr_lin * self.code_rate))

        self._awgn_noise_random = RandomState(awgn_noise_seed)
        self._wordgen_random = RandomState(wordgen_random_seed)

        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        self.gen_matrix = gen_matrix

        self.puncturing = puncturing
        self.shortening = shortening
        self.allowed_llr_range = allowed_llr_range

    def __call__(
            self,
            gentype: str="per_snr",
            *args,
            **kwargs
        ):
        if gentype == "per_snr":
            return self._gendata_per_snr(*args, **kwargs)
        elif gentype == "mix_snr":
            return self._gendata_mixed(*args, **kwargs)
        raise AttributeError("attribute `gentype` must be \"per_snr\" or \"mix_snr\".")

    def _gendata_per_snr(
            self,
            word_length: int,
            Z: int,
            is_y_all_zero: bool = True,
            decoding_type: DecoderType = DecoderType.MS,
            decoder_qms_qbit: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with all samples at the same SNR for each batch."""
        if word_length <= 0:
            raise ValueError("word_length must be positive integer")

        X = np.zeros([1, self.N * Z], dtype=self.x_dtype)
        Y = np.zeros([1, self.N * Z], dtype=self.y_dtype)

        curr_batch_size = 0
        
        for each_sf in self.snr_sigma:
            while curr_batch_size < word_length:
                # Generate codeword
                Y_i = self._gen_y(1, Z, is_y_all_zero)
                
                # BPSK modulation: 0 → +1, 1 → -1
                # Original: (-1) ** (1 - Y_i) = (-1)^1 for Y_i=0 → -1, (-1)^0 for Y_i=1 → +1
                # Which means: bit 0 → -1, bit 1 → +1 (inverted from typical)
                noise = self._awgn_noise_random.normal(0.0, 1.0, Y_i.shape)
                X_p_i = noise * each_sf + (-1) ** (1 - Y_i)
                
                x_llr_i = 2 * X_p_i / (each_sf ** 2)

                # Apply quantization for QMS
                if decoding_type == DecoderType.QMS:
                    x_llr_i = Functions.Cal_MSA_Q(x_llr_i, decoder_qms_qbit)

                # Puncturing
                if self.puncturing.start > 0:
                    if decoding_type == DecoderType.SP:
                        x_llr_i[0, self.puncturing.start - 1:self.puncturing.end] = 0.001
                    else:
                        x_llr_i[0, self.puncturing.start - 1:self.puncturing.end] = 0

                # Shortening
                if self.shortening.start > 0:
                    x_llr_i[0, self.shortening.start - 1:self.shortening.end] = -self.allowed_llr_range.abs

                X = np.vstack((X, x_llr_i))
                Y = np.vstack((Y, Y_i))
                curr_batch_size += 1
                
                if curr_batch_size == word_length:
                    break

        # Remove the initial zero row
        X = X[1:]
        Y = Y[1:]
        
        # Reshape to [batch_size, N, Z]
        X = np.reshape(X, [word_length, self.N, Z])
        
        return X, Y

    def _gendata_mixed(
            self,
            word_length: int,
            Z: int,
            is_y_all_zero: bool = True,
            decoding_type: DecoderType = DecoderType.MS,
            decoder_qms_qbit: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with mixed SNR values across the batch."""
        if word_length <= 0:
            raise ValueError("word_length must be positive integer")

        X = np.zeros([1, self.N * Z], dtype=self.x_dtype)
        Y = np.zeros([1, self.N * Z], dtype=self.y_dtype)

        curr_batch_size = 0
        
        while curr_batch_size < word_length:
            for each_sf in self.snr_sigma:
                # Generate codeword
                Y_i = self._gen_y(1, Z, is_y_all_zero)
                
                # BPSK modulation with AWGN
                noise = self._awgn_noise_random.normal(0.0, 1.0, Y_i.shape)
                X_p_i = noise * each_sf + (-1) ** (1 - Y_i)
                
                x_llr_i = 2 * X_p_i / (each_sf ** 2)

                # Apply quantization for QMS
                if decoding_type == DecoderType.QMS:
                    x_llr_i = Functions.Cal_MSA_Q(x_llr_i, decoder_qms_qbit)

                # Puncturing
                if self.puncturing.start > 0:
                    if decoding_type == DecoderType.SP:
                        x_llr_i[0, self.puncturing.start - 1:self.puncturing.end] = 0.001
                    else:
                        x_llr_i[0, self.puncturing.start - 1:self.puncturing.end] = 0

                # Shortening
                if self.shortening.start > 0:
                    x_llr_i[0, self.shortening.start - 1:self.shortening.end] = -self.allowed_llr_range.abs

                X = np.vstack((X, x_llr_i))
                Y = np.vstack((Y, Y_i))
                curr_batch_size += 1
                
                if curr_batch_size == word_length:
                    break

        # Remove the initial zero row
        X = X[1:]
        Y = Y[1:]
        
        # Reshape to [batch_size, N, Z]
        X = np.reshape(X, [word_length, self.N, Z])
        
        return X, Y

    def _gen_y(self, word_length: int, Z: int, is_y_all_zero: bool) -> np.ndarray:
        """Generate codeword (all-zero or random)."""
        if is_y_all_zero:
            return np.zeros([word_length, self.N * Z], dtype=self.y_dtype)
        else:
            if self.gen_matrix is None:
                raise ValueError("gen_matrix must be provided when is_y_all_zero is False")
            infoWord = self._wordgen_random.randint(0, 2, size=(word_length, self.K * Z))
            return np.dot(infoWord, self.gen_matrix) % 2
