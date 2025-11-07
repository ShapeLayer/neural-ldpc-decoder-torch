import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


class CheckPointUtil:
    """Utility for saving and loading model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint utility
        
        Args:
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save full checkpoint with model, optimizer, and metadata
        
        Args:
            filepath: Checkpoint filename (e.g., "checkpoint_epoch_0001.pth")
            model: PyTorch model
            optimizer: Optional optimizer
            epoch: Optional epoch number
            metrics: Optional dictionary of metrics
            config: Optional configuration dictionary
            
        Returns:
            Full path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filepath)
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint_data['epoch'] = epoch
        
        if metrics is not None:
            checkpoint_data.update(metrics)
        
        if config is not None:
            checkpoint_data['config'] = config
        
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def save_weights(
        self,
        filepath: str,
        model: torch.nn.Module,
        as_txt: bool = False
    ) -> str:
        """
        Save model weights only (no optimizer state)
        
        Args:
            filepath: Weights filename (e.g., "weights_epoch_0001")
            model: PyTorch model
            as_txt: If True, also save weights as numpy-style txt files (one file per parameter)
            
        Returns:
            Full path to saved weights file
        """
        # Save as .pth
        if not filepath.endswith('.pth'):
            pth_filepath = filepath + '.pth'
        else:
            pth_filepath = filepath
        
        weights_path = os.path.join(self.checkpoint_dir, pth_filepath)
        torch.save(model.state_dict(), weights_path)
        
        if as_txt:
            base_name = filepath.replace('.pth', '')
            txt_dir = os.path.join(self.checkpoint_dir, f"{base_name}_weights_txt")
            os.makedirs(txt_dir, exist_ok=True)
            
            index_file = os.path.join(txt_dir, "index.txt")
            with open(index_file, 'w') as f:
                f.write(f"# Model weights saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total parameters: {sum(p.numel() for p in model.parameters())}\n")
                f.write(f"# Format: Each parameter saved in separate .txt file\n")
                f.write("-" * 80 + "\n")
                f.write("Parameter_Name, Shape, Filename\n")
            
            state_dict = model.state_dict()
            for name, param in state_dict.items():
                safe_name = name.replace('.', '_').replace('/', '_')
                param_filename = f"{safe_name}.txt"
                param_path = os.path.join(txt_dir, param_filename)
                
                param_numpy = param.cpu().numpy()
                
                if param_numpy.ndim > 2:
                    original_shape = param_numpy.shape
                    param_numpy_2d = param_numpy.reshape(original_shape[0], -1)
                    np.savetxt(param_path, param_numpy_2d, 
                              header=f"Original shape: {original_shape}\nReshaped to 2D for savetxt")
                else:
                    np.savetxt(param_path, param_numpy)
                
                with open(index_file, 'a') as f:
                    f.write(f"{name}, {list(param.shape)}, {param_filename}\n")
        
        return weights_path
    
    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            filepath: Checkpoint filename or full path
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            device: Optional device to map loaded tensors to
            
        Returns:
            Checkpoint data dictionary
        """
        if not os.path.isabs(filepath):
            checkpoint_path = os.path.join(self.checkpoint_dir, filepath)
        else:
            checkpoint_path = filepath
        
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
