import os
from datetime import datetime
from typing import Dict, Any, Optional

class MetricsLogger:
    """Utility for logging training metrics to text file"""
    
    def __init__(self, log_dir: str = "checkpoints", filename: str = "training_metrics.txt"):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save log file
            filename: Log filename
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        self.best_ber = float('inf')
    
    def log(
        self,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_filename: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
            checkpoint_filename: Associated checkpoint filename
            config: Optional configuration (written as header on first call)
        """
        if epoch == 0 and config is not None:
            with open(self.log_file, 'w') as f:
                f.write(f"# Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Config: {', '.join([f'{k}={v}' for k, v in config.items()])}\n")
                f.write(f"# Columns: Epoch, Timestamp, {', '.join(metrics.keys())}, Checkpoint_File\n")
                f.write("-" * 120 + "\n")
        
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{epoch:4d}, {timestamp}, ")
            
            metric_strs = []
            for key, value in metrics.items():
                if 'ber' in key.lower():
                    metric_strs.append(f"{value:.6e}")
                else:
                    metric_strs.append(f"{value:.6f}")
            
            f.write(", ".join(metric_strs))
            f.write(f", {checkpoint_filename}\n")
    
    def is_best(self, ber: float) -> bool:
        """
        Check if current BER is the best so far
        
        Args:
            ber: Current BER value
            
        Returns:
            True if this is a new best BER
        """
        if ber < self.best_ber:
            self.best_ber = ber
            return True
        return False
