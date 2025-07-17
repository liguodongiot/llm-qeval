import torch

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)

    scales = w.abs().max(dim=-1, keepdim=True)[0]

    q_max = 2 ** (n_bits - 1) - 1

    # clamp_ 函数的功能是将张量中的每个元素的值限制在指定的最小值和最大值之间。
    # 下划线（_）表示这是一个原地操作，即直接修改原张量，而不创建新的张量。
    # div_ 函数用于将张量中的每个元素除以一个指定的值，这个操作也是原地操作。
    scales.clamp_(min=1e-5).div_(q_max)
    
    w.div_(scales).round_().mul_(scales)

    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1

    scales.clamp_(min=1e-5).div_(q_max)

    w.div_(scales).round_().mul_(scales)

    return w

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    
    t_shape = t.shape
    # 三维变二维
    t.view(-1, t_shape[-1])

    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])

    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    return t


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



