import math
import torch
import functools


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup, 
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    # _decay_func = functools.partial(
    #     _constant_warmup,
    #     warmup_iterations=T_warmup,
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
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