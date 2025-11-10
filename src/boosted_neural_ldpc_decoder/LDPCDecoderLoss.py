import torch
import torch.nn as nn


class LDPCDecoderLoss(nn.Module):
    """
    Custom loss function for LDPC decoder training
    Handles LLR output convention where positive LLR means bit=0
    
    Matches TensorFlow implementation with multi-iteration weighted loss
    """

    def __init__(
            self, 
            loss_type: int = 0,
            etha: float = 1.0,
            training_iter_start: int = 0,
            training_iter_end: int = 1,
            fixed_init: int = 0,
            fixed_iter: int = 0,
            frame_penalty_weight: float = 0.5,
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
            frame_penalty_weight: Weight for frame error penalty in loss
        """
        super(LDPCDecoderLoss, self).__init__()
        self.loss_type = loss_type
        self.etha = etha
        self.training_iter_start = training_iter_start
        self.training_iter_end = training_iter_end
        self.fixed_init = fixed_init
        self.fixed_iter = fixed_iter
        self.frame_penalty_weight = frame_penalty_weight

    def forward(self, outputs: list, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss across multiple decoder iterations
        
        Args:
            outputs: List of decoder outputs from each iteration [iter][batch, N*Z]
            targets: Target codeword bits [batch, N*Z]
            
        Returns:
            Weighted average loss across iterations
        """
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list of tensors from each iteration")
        
        # Determine iteration range
        start_t = self.training_iter_end - 1
        end_t = max(self.training_iter_start - self.fixed_init, self.fixed_iter) - 1
        
        loss_sum = 0.0
        weight_sum = 0.0
        
        # Iterate backwards from most recent to earliest iteration
        for t in range(start_t, end_t, -1):
            if t >= len(outputs):
                continue
                
            x_temp = outputs[t]  # [batch, N*Z]
            
            # Compute iteration weight
            iter_weight = self.etha ** (self.training_iter_end - 1 - t)
            
            if self.loss_type == 0:  # BCE with logits
                # Compute bit-level BCE loss
                bit_loss = nn.functional.binary_cross_entropy_with_logits(
                    -x_temp,
                    targets,
                    reduction='none'
                )  # [batch, N*Z]
                
                # Compute frame-level loss
                frame_loss = torch.mean(bit_loss, dim=1)  # [batch]
                
                # Check if frame has errors
                hard_decisions = (x_temp < 0).float()
                frame_errors = torch.any(hard_decisions != targets, dim=1).float()  # [batch]
                
                iter_loss = frame_loss + self.frame_penalty_weight * frame_errors
                
            elif self.loss_type == 1:  # Soft BER
                iter_loss = torch.mean(torch.sigmoid(x_temp), dim=1)  # [batch]
                
            elif self.loss_type == 2:  # FER
                min_llr = torch.min(-x_temp, dim=1)[0]  # [batch]
                
                sign_val = torch.sign(min_llr)
                approx = self._inv_exp(min_llr)
                sign_through = approx + (sign_val - approx).detach()
                
                iter_loss = 0.5 * (1.0 - sign_through)  # [batch]
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")

            # Accumulate weighted loss
            loss_sum = loss_sum + iter_weight * iter_loss
            weight_sum = weight_sum + iter_weight
        
        # Normalize by sum of weights
        loss_ftn = loss_sum / weight_sum if weight_sum > 0 else loss_sum
        
        # Take mean across batch
        return torch.mean(loss_ftn)

    @staticmethod
    def _inv_exp(x: torch.Tensor) -> torch.Tensor:
        """Inverse exponential approximation for sign function"""
        return 2.0 / (1.0 + torch.exp(-x)) - 1.0
