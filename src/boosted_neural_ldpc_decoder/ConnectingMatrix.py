import numpy as np

class ConnectingMatrix:
    def __init__(
            self,
            Z: int,
            basegraph: np.ndarray,
            dtype_w_odd2even=np.float32,
            dtype_w_skipconn2even=np.float32,
            dtype_w_even2odd=np.float32,
            dtype_w_even2odd_with_self=np.float32,
            dtype_w_output=np.float32,
            dtype_w_skipconn2odd=np.float32,
            dtype_lifting_matrix=np.float32
    ):
        """
        Connecting Matrix Constructor

        Keyword arguments:
        N: int -- Codeword Length
        M: int -- Number of Parity Checks
        Z: int -- Lifting Size
        sum_edge: int -- Total number of edges in the Tanner graph
        dtype_w_odd2even: np.dtype -- Data type for W_odd2even matrix (default: np.float32)
        dtype_w_skipconn2even: np.dtype -- Data type for W_skipconn2even matrix (default: np.float32)
        dtype_w_even2odd: np.dtype -- Data type for W_even2odd matrix (default: np.float32)
        dtype_w_output: np.dtype -- Data type for W_output matrix (default: np.float32)
        dtype_lifting_matrix: np.dtype -- Data type for lifting_matrix matrix (default: np.float32)

        Weights details:
        W_odd2even -- A Matrix for VN(Variable Node)
        W_even2odd -- A Matrix for CN(Parity Check Node)
        W_output -- A Matrix for output. The output would become from odd layers.
        W_skipconn2even -- A Matrix for Channel input.
        """
        self.basegraph = basegraph.copy()
        self.M, self.N = self.basegraph.shape
        self.Z = Z

        self.basegraph_binary = basegraph.copy()
        for i in range(self.M):
            for j in range(self.N):
                if self.basegraph_binary[i, j] == -1:
                    self.basegraph_binary[i, j] = 0
                else:
                    self.basegraph_binary[i, j] = 1

        self.sum_edge_c = np.sum(self.basegraph_binary, axis=1)
        self.sum_edge_v = np.sum(self.basegraph_binary, axis=0)
        self.sum_edge = np.sum(self.sum_edge_v)

        self.dtype_w_odd2even = dtype_w_odd2even
        self.dtype_w_skipconn2even = dtype_w_skipconn2even
        self.dtype_w_even2odd = dtype_w_even2odd
        self.dtype_w_even2odd_with_self = dtype_w_even2odd_with_self
        self.dtype_w_output = dtype_w_output
        self.dtype_w_skipconn2odd = dtype_w_skipconn2odd
        self.dtype_lifting_matrix = dtype_lifting_matrix

        self.neurons_per_even_layer = self.sum_edge
        self.neurons_per_odd_layer = self.sum_edge

        self.W_odd2even = np.zeros((self.sum_edge, self.sum_edge), dtype=self.dtype_w_odd2even)
        self.W_skipconn2even = np.zeros((self.N, self.sum_edge), dtype=self.dtype_w_skipconn2even)
        self.W_even2odd = np.zeros((self.sum_edge, self.sum_edge), dtype=self.dtype_w_even2odd)
        self.W_even2odd_with_self = np.zeros((self.sum_edge, self.sum_edge), dtype=self.dtype_w_even2odd_with_self)
        self.W_output = np.zeros((self.sum_edge, self.N), dtype=self.dtype_w_output)
        self.W_skipconn2odd = np.zeros((self.M, self.sum_edge), dtype=self.dtype_w_skipconn2odd)

        self.lifting_matrix_1: np.ndarray = np.zeros((self.neurons_per_odd_layer * self.Z, self.neurons_per_odd_layer * self.Z),
                                   self.dtype_lifting_matrix)
        self.lifting_matrix_2: np.ndarray = np.zeros((self.neurons_per_odd_layer * self.Z, self.neurons_per_odd_layer * self.Z),
                                   self.dtype_lifting_matrix)

        self._init_conn_matrix()

    def _init_conn_matrix(self):
        # lifting_matrix
        k = 0
        for j in range(self.basegraph.shape[1]):
            for i in range(self.basegraph.shape[0]):
                if self.basegraph[i, j] != -1:
                    _lifted_num = self.basegraph[i, j] % self.Z
                    for h in range(self.Z):
                        self.lifting_matrix_1[k * self.Z + h, k * self.Z + (h + _lifted_num) % self.Z] = 1
                    k += 1
        k = 0
        for i in range(self.basegraph.shape[0]):
            for j in range(self.basegraph.shape[1]):
                if self.basegraph[i, j] != -1:
                    _lifted_num = self.basegraph[i, j] % self.Z
                    for h in range(self.Z):
                        self.lifting_matrix_2[k * self.Z + h, k * self.Z + (h + _lifted_num) % self.Z] = 1
                    k += 1

        # W_odd2even
        k = 0
        for j in range(self.N):  # run over the columns
            for i in range(self.M):  # break after the first one
                if self.basegraph_binary[i, j] == 1:  # finding the first one is ok
                    num_of_conn = int(
                        np.sum(self.basegraph_binary[:, j]))  # get the number of connection of the variable node
                    idx = np.argwhere(self.basegraph_binary[:, j] == 1)  # get the indexes
                    for l in range(num_of_conn):  # adding num_of_conn columns to W
                        vec_tmp = np.zeros((self.sum_edge), dtype=self.dtype_w_odd2even)
                        for r in range(self.M):  # adding one to the right place
                            if self.basegraph_binary[r, j] == 1 and idx[l][0] != r:
                                idx_row = np.cumsum(self.basegraph_binary[r, 0:j + 1])[-1] - 1
                                odd_layer_node_count = 0
                                if r > 0:
                                    odd_layer_node_count = np.cumsum(self.sum_edge_c[0:r])[-1]
                                vec_tmp[idx_row + odd_layer_node_count] = 1  # offset index adding
                        self.W_odd2even[:, k] = vec_tmp.transpose()
                        k += 1
                    break

        # W_even2odd
        k = 0
        for j in range(self.N):
            for i in range(self.M):
                if self.basegraph_binary[i, j] == 1:
                    idx_row = np.cumsum(self.basegraph_binary[i, 0:j + 1])[-1] - 1
                    idx_col = np.cumsum(self.basegraph_binary[0: i + 1, j])[-1] - 1
                    odd_layer_node_count_1 = 0
                    odd_layer_node_count_2 = np.cumsum(self.sum_edge_c[0:i + 1])[-1]
                    if i > 0:
                        odd_layer_node_count_1 = np.cumsum(self.sum_edge_c[0:i])[-1]
                    self.W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                    self.W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                    self.W_even2odd_with_self[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                    k += 1  # k is counted in column direction

        # W_output
        k = 0
        for j in range(self.N):
            for i in range(self.M):
                if (self.basegraph_binary[i, j] == 1):
                    idx_row = np.cumsum(self.basegraph_binary[i, 0:j + 1])[-1] - 1
                    idx_col = np.cumsum(self.basegraph_binary[0: i + 1, j])[-1] - 1
                    odd_layer_node_count = 0
                    if i > 0:
                        odd_layer_node_count = np.cumsum(self.sum_edge_c[0:i])[-1]
                    self.W_output[odd_layer_node_count + idx_row, k] = 1.0
            k += 1

        # W_skipconn2even
        k = 0
        for j in range(self.N):
            for i in range(self.M):
                if (self.basegraph_binary[i, j] == 1):
                    self.W_skipconn2even[j, k] = 1.0
                    k += 1
        
        # W_skipconn2odd
        k = 0
        for i in range(self.M):
            for j in range(self.N):
                if (self.basegraph_binary[i, j] == 1):
                    self.W_skipconn2odd[i, k] = 1.0
                    k += 1
