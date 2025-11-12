from typing import Optional
import torch
import torch.nn as nn
from boosted_neural_ldpc_decoder.struct.LossType import LossType
from boosted_neural_ldpc_decoder.Functions import Functions

class LDPCDecoderLoss(nn.Module):
    """
    Custom loss function for LDPC decoder training
    Handles LLR output convention where positive LLR means bit=0
    
    Matches TensorFlow implementation with multi-iteration weighted loss
    """

    def __init__(
            self, 
            loss_type: LossType = LossType.BCE,
            etha: float = 1.0,
    ):
        """
        Initialize loss function
        
        Args:
            loss_type: Type of loss
                0 = BCE with logits (sigmoid_cross_entropy_with_logits)
                1 = Soft BER (sigmoid-based, for all-zero codeword)
                2 = FER (frame error rate based, for all-zero codeword)
            etha: Exponential weight decay factor for multi-iteration loss
            training_iter_start: Start iteration for training
            training_iter_end: End iteration for training
            fixed_init: Fixed initialization iterations (Delta_2)
            fixed_iter: Fixed iterations before training starts
        """
        super(LDPCDecoderLoss, self).__init__()
        self.loss_type = loss_type
        self.etha = etha

    def forward(
            self,
            outputs: Optional[list | torch.Tensor],
            targets: Optional[list | torch.Tensor],
            coeff_param: Optional[list | int] = 1,
        ) -> torch.Tensor:
        # Available Cases in outputs and targets:
        is_valid = 0
        # 1. outputs: torch.Tensor, targets: torch.Tensor
        if is_valid == 0 and (isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor)):
            is_valid = 1
        # 2. outputs: list of torch.Tensor, targets: torch.Tensor
        if is_valid == 0 and (isinstance(outputs, list) and isinstance(targets, torch.Tensor)):
            is_valid = 2
        # 3. outputs: list of torch.Tensor, targets: list of torch.Tensor
        if not is_valid and isinstance(outputs, list) and isinstance(targets, list):
            if len(outputs) == len(targets):
                is_valid = 3
        if is_valid == 0:
            raise ValueError("Invalid types for outputs and targets in LDPCDecoderLoss. Outputs must be either a torch.Tensor or a list of torch.Tensor. Targets must be either a torch.Tensor or a list of torch.Tensor with matching length to outputs.")
        
        # Validate Coeff Param used in weighting loss across iterations
        if is_valid == 1:
            if not isinstance(coeff_param, int):
                raise ValueError("Invalid coeff_param provided to LDPCDecoderLoss. Must be an integer when outputs is a single torch.Tensor.")
        else:
            if not (
                (isinstance(coeff_param, list) and len(coeff_param) == len(outputs)),
                isinstance(coeff_param, int)
            ):
                raise ValueError("Invalid coeff_param provided to LDPCDecoderLoss. Must be either a list of integers matching outputs length or a single integer.")

        loss_ftn = 0
        _coeff = 0
        
        for curr_iter in range(
            len(outputs) if is_valid != 1 else 1
        ):
            now_actual = outputs[curr_iter] if is_valid != 1 else outputs
            now_target = targets[curr_iter] if is_valid == 3 else targets
            now_coeff = 1
            if coeff_param is not None:
                now_coeff = coeff_param[curr_iter] if isinstance(coeff_param, list) else coeff_param

            if self.loss_type == LossType.BCE:
                loss_ftn = loss_ftn + pow(
                    self.etha,
                    now_coeff
                ) * nn.functional.binary_cross_entropy_with_logits(
                    now_actual,  # logits
                    now_target,  # labels
                )
            elif self.loss_type == LossType.SoftBEROnAllZero:
                loss_ftn = loss_ftn + pow(
                    self.etha,
                    now_coeff
                ) * torch.sigmoid(now_actual)
            elif self.loss_type == LossType.FEROnAllZero:
                _now_actual = 1 / 2 * (1 - Functions.sign_through_torch(torch.min(-now_actual, dim=1)[0]))
                loss_ftn = loss_ftn + pow(
                    self.etha,
                    now_coeff
                ) * _now_actual
            
            _coeff = _coeff + pow(
                self.etha,
                now_coeff
            )

        loss_ftn = loss_ftn / _coeff if _coeff > 0 else loss_ftn
        return 1.0 * loss_ftn.mean()
