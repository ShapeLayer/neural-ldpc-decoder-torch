from boosted_neural_ldpc_decoder.struct.NodeType import NodeType

class NodeWeightSharingConfig:
    def __init__(
            self,
            cn_weight_sharing: int,
            ucn_weight_sharing: int,
            vn_weight_sharing: int,
    ):
        self.cn_weight_sharing = cn_weight_sharing
        self.ucn_weight_sharing = ucn_weight_sharing
        self.vn_weight_sharing = vn_weight_sharing

    def __iter__(self):
        yield (NodeType.CN, self.cn_weight_sharing)
        yield (NodeType.UCN, self.ucn_weight_sharing)
        yield (NodeType.VN, self.vn_weight_sharing)

    def get(
            self,
            node_type: NodeType,
    ):
        if node_type == NodeType.CN:
            return self.cn_weight_sharing
        elif node_type == NodeType.UCN:
            return self.ucn_weight_sharing
        elif node_type == NodeType.VN:
            return self.vn_weight_sharing
