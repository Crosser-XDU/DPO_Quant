from typing import Dict, Union
import torch

def add_eos(token: Dict, eos_token_id: int):
    """Add an EOS token to a token dictionary."""
    token['input_ids'].append(eos_token_id)
    token['attention_mask'].append(1)

def move_to_device(batch: Dict, device: int) -> Dict:
    """Move a batch to the specified device."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def get_micro_batch(batch: Dict, micro_batch_index: int, micro_batch_size: int, device: int) -> Dict:
    """Get a micro batch from a batch and move it to the specified device."""
    start = micro_batch_index * micro_batch_size
    end = (micro_batch_index + 1) * micro_batch_size
    micro_batch = {k: v[start:end] for k, v in batch.items()}
    return micro_batch

def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """Pad a tensor to a specified length."""
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0