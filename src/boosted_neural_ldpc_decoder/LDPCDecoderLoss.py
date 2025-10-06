import torch
import torch.nn as nn


class LDPCDecoderLoss(nn.Module):
    """
    Custom loss function for LDPC decoder training
    """

    def __init__(self, loss_type=0, iter_weights=None):
        """
        Initialize loss function
        
        Args:
            loss_type: Type of loss (0=BCE, 1=MSE, 2=Weighted BCE)
            iter_weights: Optional weights for each iteration
        """
        super(LDPCDecoderLoss, self).__init__()
        self.loss_type = loss_type
        self.iter_weights = iter_weights

    def forward(self, outputs, targets, iteration_idx=None):
        """
        Compute loss for decoder outputs
        
        Args:
            outputs: List of outputs or single output tensor
            targets: Target values
            iteration_idx: If provided, compute loss only for this iteration
            
        Returns:
            Loss value
        """
        if isinstance(outputs, list):
            if iteration_idx is not None:
                # Use only the specified iteration
                output = outputs[iteration_idx]
                return self._compute_single_loss(output, targets, iteration_idx)
            else:
                # Compute loss for all iterations
                total_loss = 0
                for i, output in enumerate(outputs):
                    loss = self._compute_single_loss(output, targets, i)
                    total_loss += loss
                return total_loss / len(outputs)
        else:
            # Single output tensor
            return self._compute_single_loss(outputs, targets)

    def _compute_single_loss(self, output, targets, iter_idx=0):
        """Compute loss for a single output tensor"""
        if self.loss_type == 0:  # BCE with logits
            loss = nn.functional.binary_cross_entropy_with_logits(
                output, targets, reduction='mean'
            )
        elif self.loss_type == 1:  # MSE
            loss = nn.functional.mse_loss(
                output, targets, reduction='mean'
            )
        elif self.loss_type == 2:  # Weighted BCE
            # Weight based on iteration if available
            weight = 1.0
            if self.iter_weights:
                weight = self.iter_weights[iter_idx]
            else:
                # Default increasing weight by iteration
                weight = 1.0 + 0.1 * iter_idx

            loss = weight * nn.functional.binary_cross_entropy_with_logits(
                output, targets, reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=output.device)

        return loss
