# %%
# 3.1s
from utils import build_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quant import quant, dequant, quant_unpack, dequant_unpack
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
import functools
from functools import partial
import torch
from tqdm import tqdm

# %%
model_name = 'facebook/opt-6.7b'
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

# %%
bits = 8
gs = 128
scale, zero, qs = quant(bits, gs, model)
q_x = dequant(scale, zero, qs, gs, bits)

# %%
model.eval()
device = next(model.parameters()).device # next는 객체의 __next__ 호출, 다음 iter를 부름?
act_scales = {}
scaled_act_max = {}
scaled_act_min = {}
scaled_act_sum = {}
scaled_act_absum = {}
scaled_act_numel = {}

def stat_tensor(name, tensor):
    hidden_dim = tensor.shape[-1]
    tensor = tensor.view(-1, hidden_dim).detach()
    act_shape = tensor.shape
    for i in range(len(scale[name])):
        temp = scale[name][i].expand(gs, -1).T.flatten().expand(act_shape[0], -1).to(device)
        scaled_act = (tensor * temp)
        if name in scaled_act_max:
            scaled_act_max[name] = scaled_act.max() if scaled_act_max[name] < scaled_act.max() else scaled_act_max[name]
            scaled_act_min[name] = scaled_act.min() if scaled_act_min[name] > scaled_act.min() else scaled_act_min[name]
            scaled_act_sum[name] += scaled_act.sum()
            scaled_act_absum[name] += torch.abs(scaled_act).sum()
            scaled_act_numel[name] += torch.numel(scaled_act)
        else:
            scaled_act_max[name] = scaled_act.max()
            scaled_act_min[name] = scaled_act.min()
            scaled_act_sum[name] = scaled_act.sum()
            scaled_act_absum[name] = torch.abs(scaled_act).sum()
            scaled_act_numel[name] = torch.numel(scaled_act)


def stat_input_hook(m, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    stat_tensor(name, x)


# %%
hooks = []
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) | isinstance(m, Conv1D):
        if name.split('.')[-1] != 'lm_head':
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
        

# %%
dataset = dataset.shuffle(seed=42)
dataset_list = []
for ii in range(len(dataset)):
    if dataset[ii]['text'] != '':
        dataset_list.append(dataset[ii])


# %%
num_samples = 4
seq_len = 64

for i in tqdm(range(num_samples)):
    input_ids = tokenizer(dataset_list[i]["text"], return_tensors="pt",
                            max_length=seq_len, truncation=True).input_ids.to(device)
    model(input_ids)

for h in hooks:
    h.remove()

# %%
scaled_act_avg = {}
scaled_act_absavg = {}
scaled_act_minmax = {}
for key in scaled_act_max:
    scaled_act_avg[key] = scaled_act_sum[key] / scaled_act_numel[key]
    scaled_act_absavg[key] = scaled_act_absum[key] / scaled_act_numel[key]
    scaled_act_minmax[key] = scaled_act_max[key] - scaled_act_min[key]

# %%
torch.save(scaled_act_max, '/raid/jwjeong/results/scaled_act_max_opt.pt')
torch.save(scaled_act_min, '/raid/jwjeong/results/scaled_act_min_opt.pt')
torch.save(scaled_act_minmax, '/raid/jwjeong/results/scaled_act_minmax_opt.pt')
torch.save(scaled_act_avg, '/raid/jwjeong/results/scaled_act_avg_opt.pt')
torch.save(scaled_act_absavg, '/raid/jwjeong/results/scaled_act_absavg_opt.pt')


