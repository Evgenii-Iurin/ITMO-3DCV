import os
import torch
from datetime import datetime


def save_trained_model(model, optimizer=None, loss=None, save_dir="checkpoints", filename=None):
    """
    Save an already trained model
    
    Args:
        model: The trained NeRF model to save
        optimizer: Optional optimizer state to save
        loss: Optional loss value to record
        save_dir: Directory to save the model to
        filename: Optional specific filename, otherwise auto-generated
        
    Returns:
        Path to the saved model file
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Auto-generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nerf_model_{timestamp}.pth"
    
    # Construct full path
    filepath = os.path.join(save_dir, filename)
    
    # Build checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'embedding_dim_pos': model.embedding_dim_pos,
        'embedding_dim_direction': model.embedding_dim_direction,
    }
    
    # Add optimizer if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add loss if provided
    if loss is not None:
        checkpoint['loss'] = loss
        
    # Save to disk
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    
    return filepath

def load_trained_model(filepath, model, device='cuda'):
    """
    Load a saved NeRF model
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model to
        
    Returns:
        Loaded NeRF model
    """
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create a new model with the same parameters
    model = model(
        embedding_dim_pos=checkpoint.get('embedding_dim_pos', 30),
        embedding_dim_direction=checkpoint.get('embedding_dim_direction', 4)
    ).to(device)
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode for inference
    model.eval()
    
    return model



def load_checkpoint(filepath, model_class, model=None, optimizer=None, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        model: Optional model to load weights into (will create new if None)
        optimizer: Optional optimizer to load state into
        device: Device to load model to
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer (if provided)
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    # Load checkpoint from file
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model if not provided
    if model is None:
        model = model_class(
            embedding_dim_pos=checkpoint.get('embedding_dim_pos', 30),
            embedding_dim_direction=checkpoint.get('embedding_dim_direction', 4)
        ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get additional info
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return model, optimizer, epoch, loss