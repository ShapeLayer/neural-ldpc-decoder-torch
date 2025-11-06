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
        self.training_iter_start = training_iter_start
        self.training_iter_end = training_iter_end
        self.fixed_init = fixed_init
        self.fixed_iter = fixed_iter

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
        
        # Determine iteration range: from training_iter_end-1 down to max(training_iter_start-fixed_init, fixed_iter)
        start_t = self.training_iter_end - 1
        end_t = max(self.training_iter_start - self.fixed_init, self.fixed_iter) - 1
        
        loss_sum = 0.0
        weight_sum = 0.0
        
        # Iterate backwards from most recent to earliest iteration
        for t in range(start_t, end_t, -1):
            if t >= len(outputs):
                continue
                
            x_temp = outputs[t]  # [batch, N*Z]
            
            # Compute iteration weight: etha^(training_iter_end - 1 - t)
            iter_weight = self.etha ** (self.training_iter_end - 1 - t)
            
            # Compute loss for this iteration
            if self.loss_type == 0:  # BCE with logits
                # TF: sigmoid_cross_entropy_with_logits(labels=ya, logits=x_temp)
                # PyTorch BCE expects: positive logit = class 1
                # Our LLR: positive = bit 0, negative = bit 1
                # So we DON'T negate here because TF also uses raw logits
                iter_loss = nn.functional.binary_cross_entropy_with_logits(
                    -x_temp,
                    targets,
                    reduction='none'  # Keep per-sample loss for weighting
                )
                
            elif self.loss_type == 1:  # Soft BER (all-zero codeword only)
                # TF: tf.math.sigmoid(x_temp)
                # For all-zero: positive LLR (wrong prediction) should have high loss
                iter_loss = torch.sigmoid(x_temp)
                
            elif self.loss_type == 2:  # FER (all-zero codeword only)
                # TF: x_temp = 1/2*(1-sign_through(tf.reduce_min(-x_temp, axis=1)))
                min_llr = torch.min(-x_temp, dim=1)[0]  # [batch]
                
                # sign_through with straight-through estimator
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
        
        # Take mean across batch (matches TF: tf.reduce_mean(loss_ftn))
        return torch.mean(loss_ftn)

    @staticmethod
    def _inv_exp(x: torch.Tensor) -> torch.Tensor:
        """Inverse exponential approximation for sign function"""
        return 2.0 / (1.0 + torch.exp(-x)) - 1.0
