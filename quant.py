from auto_gptq.quantization.quantizer import Quantizer
from auto_gptq.quantization.gptq import GPTQ
import torch
from utils import find_layers

def quant(bits, gs, model):
    layers = find_layers(model)
    group_size = gs
    quantizer = Quantizer()
    quantizer.configure(
        bits = bits,
        perchannel=True,
        sym=False,
    )
    scales = {}
    zeros = {}
    g_idx = {}
    qs = {}
    qweight = {}
    for layer in layers.keys():
        if layer.split('.')[-1] != 'lm_head':
            module = layers[layer].weight.data
            row = module.size(0)
            col = module.size(1)
            scales[layer] = torch.zeros(row, col // group_size)
            zeros[layer] = torch.zeros(row, col // group_size)
            qs[layer] = torch.zeros((row, col), dtype=torch.int32)
            qweight[layer] = torch.zeros((qs[layer].shape[0]//32*bits, qs[layer].shape[1]), dtype=torch.int32)
            #g_idx[layer] = torch.zeros_like(zeros[layer])
            for i in range(0, col, group_size):
                x = module[:, i:i+group_size]
                quantizer.find_params(x, weight=True)
                maxq = quantizer.maxq
                scale = quantizer.scale
                zero = quantizer.zero
                q = torch.clamp(torch.round(x/scale) + zero, 0, maxq)
                qs[layer][:, i:i+group_size] = q
                scales[layer][:, i//group_size] = scale.squeeze()
                zeros[layer][:, i//group_size] = zero.squeeze()
                #g_idx[layer][:, i//group_size] = i//group_size
            
            i = 0
            row = 0
            while row < qweight[layer].shape[0]:
                if bits in [2, 4, 8, 16]:
                    for j in range(i, i + (32 // bits)):
                        qweight[layer][row] |= qs[layer][j] << (bits * (j - i))
                    i += 32 // bits
                    row += 1
    return scales, zeros, qweight

def dequant(scales, zeros, qs, gs, bits):
    q_x = {}
    group_size = gs
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    for key in qs.keys():
        temp = []
        scale = scales[key]
        zero = zeros[key]
        q = qs[key]
        row = qs[key].shape[0] * (32//bits)
        col = qs[key].shape[1]
        q_x[key] = torch.zeros(row, col, dtype=torch.int32)
        q_x[key] = torch.bitwise_right_shift(
            torch.unsqueeze(q, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int32)
        q_x[key] = torch.bitwise_and(q_x[key], (2**bits) - 1)
        q_x[key] = q_x[key].reshape(-1, q_x[key].shape[2])
        
        for i in range(0, col, group_size):
            g_idx = i // group_size
            temp.append(scale[:, g_idx].unsqueeze(1).to(q.device)*(q_x[key][:, i:i+group_size] - zero[:, g_idx].unsqueeze(1).to(q.device)))
            
        q_x[key] = torch.concat(temp, dim=1)

    return q_x

def quant_unpack(bits, gs, model):
    from utils import find_layers
    layers = find_layers(model)
    group_size = gs
    quantizer = Quantizer()
    quantizer.configure(
        bits = bits,
        perchannel=True,
        sym=False,
    )
    scales = {}
    zeros = {}
    qs = {}
    for layer in layers.keys():
        module = layers[layer].weight.data
        row = module.size(0)
        col = module.size(1)
        scales[layer] = torch.zeros(row, col // group_size)
        zeros[layer] = torch.zeros(row, col // group_size)
        qs[layer] = torch.zeros((row, col), dtype=torch.int32)
        for i in range(0, col, group_size):
            x = module[:, i:i+group_size]
            quantizer.find_params(x, weight=True)
            maxq = quantizer.maxq
            scale = quantizer.scale
            zero = quantizer.zero
            q = torch.clamp(torch.round(x/scale) + zero, 0, maxq)
            qs[layer][:, i:i+group_size] = q
            scales[layer][:, i//group_size] = scale.squeeze()
            zeros[layer][:, i//group_size] = zero.squeeze()
        
    return scales, zeros, qs

def dequant_unpack(scales, zeros, qs, gs):
    group_size = gs
    q_x = {}
    for key in qs.keys():
        temp = []
        scale = scales[key]
        zero = zeros[key]
        q = qs[key]
        col = qs[key].shape[1]
        q_x[key] = torch.zeros(q.size(), dtype=torch.float)
        
        for i in range(0, col, group_size):
            g_idx = i // group_size
            temp.append(scale[:, g_idx].unsqueeze(1).to(q.device)*(q[:, i:i+group_size] - zero[:, g_idx].unsqueeze(1).to(q.device)))
            
        q_x[key] = torch.concat(temp, dim=1)

    return q_x