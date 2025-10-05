from typing import Any

import numpy as np
from numpy import ndarray, dtype
from numpy.random import RandomState

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

        awgn_noise_seed: int=2042,
        wordgen_random_seed: int=1074,

        x_dtype=np.float32,
        y_dtype=np.int64,

        gen_matrix: np.ndarray=None,
    ):
        self.N = N
        self.M = M
        self.K = N - M
        self.snr_db = snr_db
        self.code_rate = 1.0 * (N - M) / (N - 2)
        self.snr_lin = 10.0 ** (self.snr_db / 10.0)
        self.snr_sigma = np.sqrt(1.0 / (2.0 * self.snr_lin * self.code_rate))

        self._awgn_noise_random = RandomState(awgn_noise_seed)
        self._wordgen_random = RandomState(wordgen_random_seed)

        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        self.gen_matrix = gen_matrix

    def __call__(self, *args, **kwargs):
        return self._gendata(*args, **kwargs)

    def _gendata(
        self,
        word_length: int,
        Z: int,
        is_y_all_zero: bool=True,
    ) -> tuple[list[ndarray[tuple[Any, ...], dtype[Any]]], list[ndarray[tuple[Any, ...], dtype[Any]]]]:
        """
        Generates origin bits and AWGN-Passed LLR set for training or testing a neural network decoder.

        :param word_length:
        :param Z:
        :param is_y_all_zero:
        :return:
        """
        if word_length <= 0:
            raise ValueError("word_length must be positive integer")

        x: list[np.ndarray] = []
        y: list[np.ndarray] = []

        gen_x = self._gen_x
        gen_y = self._gen_y_all_zero if is_y_all_zero else self._gen_y_wordgen

        for each_sf in self.snr_sigma:
            y_i = gen_y(word_length, Z)

            x_p_i = gen_x(word_length) * each_sf + -1 ** (1 - y_i)
            x_llr_i = 2 * x_p_i / (each_sf ** 2)
            x_llr_i = x_llr_i.astype(self.x_dtype)

            x.append(x_llr_i)
            y.append(y_i)

        return x, y

    def _gen_x(self, word_length: int) -> np.ndarray:
        return self._awgn_noise_random.normal(0., 1., size=(word_length, self.gen_matrix.shape[1])).astype(self.x_dtype)

    def _gen_y_all_zero(self, word_length: int, Z: int) -> np.ndarray:
        return np.dot(
            np.zeros(shape=(word_length, self.K * Z), dtype=self.y_dtype),
            self.gen_matrix
        ) % 2

    def _gen_y_wordgen(self, word_length: int, Z: int) -> np.ndarray:
        if self.gen_matrix is None:
            raise ValueError("self.gen_matrix must be provided when is_y_all_zero is False")
        return np.dot(
            self._wordgen_random.randint(0, 2, size=(word_length, self.K * Z)).astype(self.y_dtype),
            self.gen_matrix
        ) % 2
