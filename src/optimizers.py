import math
import torch
import functools


def cosine_decay_warmup(iteration, warmup_iterations, total_iterations, min_factor=0.0):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to min_factor
    
    Args:
        iteration: Current iteration
        warmup_iterations: Number of warmup iterations
        total_iterations: Total number of iterations
        min_factor: Minimum multiplier to decay to (default: 0.0)
    
    Returns:
        lr multiplier between min_factor and 1.0
    """
    if  warmup_iterations > 0 and iteration <= warmup_iterations:
        # Linear warmup phase
        multiplier = iteration / warmup_iterations
    else:
        # Cosine decay phase
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        # Cosine annealing from 1.0 to min_factor
        cosine_decay = 0.5 * (1 + math.cos(math.pi * multiplier))
        multiplier = min_factor + (1.0 - min_factor) * cosine_decay
    
    return multiplier

def constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup, min_lr=0.0, base_lr=None):
    """
    Creates a cosine annealing scheduler with warmup and minimum learning rate.
    
    Args:
        optimizer: PyTorch optimizer
        T_max: Total number of iterations
        T_warmup: Number of warmup iterations
        min_lr: Minimum learning rate (default: 0.0)
        base_lr: Base learning rate. If None, uses the optimizer's initial lr
                 (default: None)
    
    Returns:
        PyTorch LR scheduler
    """
    # Determine the base learning rate
    if base_lr is None:
        base_lr = optimizer.param_groups[0]['lr']
    
    # Calculate min_factor (as a proportion of base_lr)
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0
    
    decay_func = functools.partial(
        cosine_decay_warmup,
        warmup_iterations=T_warmup, 
        total_iterations=T_max,
        min_factor=min_factor
    )
    
    # Alternative constant scheduler if needed
    # decay_func = functools.partial(
    #     constant_warmup,
    #     warmup_iterations=T_warmup,
    # )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_func)
    return scheduler


def _step_decay_warmup(iteration, warmup_iterations, step_size, gamma):
    """
    Linear warmup from 0 --> 1.0, then constant, followed by step decay.
    If warmup_iterations == 0, skips warmup and applies step decay directly.
    """
    if warmup_iterations > 0 and iteration <= warmup_iterations:
        # Warmup phase
        multiplier = iteration / warmup_iterations
        return multiplier * gamma ** (iteration // step_size)
    else:
        # After warmup or if warmup is not applied
        multiplier = 1.0
    return multiplier * gamma ** ((iteration - warmup_iterations) // step_size)

def StepLRWarmup(optimizer, T_max=100, gamma=0.1, T_warmup=10):
    """
    Applies StepLR with warmup. If T_warmup=0, applies StepLR directly without warmup.
    """
    # Adjust step_size to half of T_max, or another value if needed
    step_size = T_max // 2

    # Set decay function
    _decay_func = functools.partial(
        _step_decay_warmup,
        warmup_iterations=T_warmup,
        step_size=step_size,
        gamma=gamma
    )

    # LambdaLR applies the decay function at each iteration
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    
    return scheduler

def ConstantLR(optimizer):
    # The lambda function always returns 1, so the learning rate remains constant.
    _constant_func = lambda epoch: 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _constant_func)
    return scheduler