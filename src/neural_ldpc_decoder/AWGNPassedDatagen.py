import numpy as np
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
        snr_db: list[float]=[2., 2.5, 3., 3.5, 4.],

        awgn_noise_seed: int=2042,
        wordgen_random_seed: int=1074,

        x_dtype=np.float32,
        y_dtype=np.int64,

        gen_matrix: np.ndarray=None,
    ):
        self._awgn_noise_random = RandomState(awgn_noise_seed)
        self._wordgen_random = RandomState(wordgen_random_seed)

        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        self._snr_db = snr_db

        self.gen_matrix = gen_matrix

    def __call__(self, *args, **kwargs):
        return self._gendata(*args, **kwargs)

    def _gendata(
        self,
        word_length,
        N,
        K,
        Z,
        scaling_factor: list=[16],
        is_y_all_zero: bool=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates origin bits and AWGN-Passed LLR set for training or testing a neural network decoder.

        :param word_length:
        :param N:
        :param K:
        :param Z:
        :param scaling_factor:
        :param is_y_all_zero:
        :return:
        """
        if word_length <= 0:
            raise ValueError("word_length must be positive integer")

        x = np.empty(shape=(0, N * Z), dtype=self.x_dtype)
        y = np.empty(shape=(0, N * Z), dtype=self.y_dtype)

        gen_y = self._gen_y_all_zero if is_y_all_zero else self._gen_y_wordgen

        for each_sf in scaling_factor:
            if is_y_all_zero:
                y_i = self._gen_y_all_zero(word_length, N, Z)
            else:
                y_i = gen_y(word_length, K, Z)

            x_p_i = self._gen_x(word_length, N, Z) * each_sf + -1 ** (1 - y_i)
            x_llr_i = 2 * x_p_i / (each_sf ** 2)
            x_llr_i = x_llr_i.astype(self.x_dtype)

            x = np.concatenate((x, x_llr_i))
            y = np.concatenate((y, y_i))

        return x, y

    def _gen_x(self, word_length: int, K: int, Z: int) -> np.ndarray:
        return self._awgn_noise_random.normal(0., 1., size=(word_length, K * Z)).astype(self.x_dtype)

    def _gen_y_all_zero(self, word_length: int, N: int, Z: int) -> np.ndarray:
        return np.zeros(shape=(word_length, N * Z), dtype=self.y_dtype)

    def _gen_y_wordgen(self, word_length: int, K: int, Z: int) -> np.ndarray:
        if self.gen_matrix is None:
            raise ValueError("self.gen_matrix must be provided when is_y_all_zero is False")
        return np.dot(self._wordgen_random.randint(0, 2, size=(word_length, K * Z)).astype(self.y_dtype), self.gen_matrix) % 2
