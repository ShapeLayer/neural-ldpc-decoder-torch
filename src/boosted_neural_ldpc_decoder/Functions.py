import numpy as np
import torch

class Functions:
    @staticmethod
    def hard_sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)

    @staticmethod
    def proxy_sign_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -1.0, 1.0)

    @staticmethod
    def inv_exp_torch(x: torch.Tensor) -> torch.Tensor:
        return 2.0 / (1.0 + torch.exp(-x)) - 1.0

    @staticmethod
    def round_through_torch(x: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(x)
        return Functions.hard_sigmoid_torch(x) + (rounded - Functions.hard_sigmoid_torch(x)).detach()

    @staticmethod
    def sign_through_torch(x: torch.Tensor) -> torch.Tensor:
        sign_v = torch.sign(x)
        approx = Functions.inv_exp_torch(x)
        return approx + (sign_v - approx).detach()

    @staticmethod
    def qms_clipping_torch(x: torch.Tensor, q_bit: int) -> torch.Tensor:
        if q_bit == 6:
            return torch.clamp(x, -15.5, 15.5)
        if q_bit == 5:
            return torch.clamp(x, -7.5, 7.5)
        if q_bit == -5:
            return torch.clamp(x, -15.0, 15.0)
        if q_bit == 4:
            return torch.clamp(x, -7.0, 7.0)
        if q_bit == 3:
            return torch.clamp(x, -6.0, 6.0)
        return x

    @staticmethod
    def cal_msa_q_torch(x: torch.Tensor, q_bit: int) -> torch.Tensor:
        def _quantize_for_qms(x: torch.Tensor, q_bit: int) -> torch.Tensor:
            if q_bit == 6:
                return torch.clamp(torch.round(x), -15.5, 15.5)
            if q_bit == 5:
                return torch.clamp(torch.round(x * 2.0) / 2.0, -7.5, 7.5)
            if q_bit == -5:
                return torch.clamp(torch.round(x), -15.0, 15.0)
            if q_bit == 4:
                return torch.clamp(torch.round(x), -7.0, 7.0)
            if q_bit == 3:
                return torch.clamp(torch.round(x / 2.0) * 2.0, -6.0, 6.0)
            return x
        clipped = Functions.qms_clipping_torch(x, q_bit)
        q_value = _quantize_for_qms(x, q_bit)
        return clipped + (q_value - clipped).detach()

    @staticmethod
    def Cal_MSA_Q(x, q_bit):
        if q_bit == 6:
            q_value = np.clip(np.round(x), -15.5, 15.5)
        elif q_bit == 5:
            q_value = np.clip(np.round(x * 2) / 2, -7.5, 7.5)
        elif q_bit == -5:
            q_value = np.clip(np.round(x), -15, 15)
        elif q_bit == 4:
            q_value = np.clip(np.round(x), -7, 7)
        elif q_bit == 3:
            q_value = np.clip(np.round(x / 2) * 2, -6, 6)
        else:
            q_value = x
        return q_value

    @staticmethod
    def calc_ber_fer(pred: np.ndarray, y: np.ndarray):
        # TODO
        errors = np.sum(np.any(pred != y, axis=1))
        total_bits = np.prod(y.shape)
        total_frames = y.shape[0]
        ber = np.sum(pred != y) / total_bits
        fer = errors / total_frames
        return ber, fer, errors
