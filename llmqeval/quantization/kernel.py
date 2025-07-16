import torch

def absmax_int8(X):
    scale = 127 / torch.max(torch.abs(X))  # Adjusted scale
    X_quant = (scale * X).round()
    
    X_dequant = X_quant / scale
    return X_quant.to(torch.int8), X_dequant


def zeropoint_int8(X):
    x_range = torch.max(X) - torch.min(X)

    x_range = 1 if x_range == 0 else x_range
    
    scale = 255 / x_range

    zeropoint = (-scale * torch.min(X)).round() - 128

    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    X_dequant = (X_quant - zeropoint) / scale
    return X_quant.to(torch.int8), X_dequant
