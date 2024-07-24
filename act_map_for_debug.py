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
    act_shape = tensor.shape
    hidden_dim = tensor.shape[-1]
    tensor = tensor.view(-1, hidden_dim).abs().detach()

    for i in range(len(scale[name].T)):
        temp = scale[name].T[i].expand(gs, -1).flatten().expand(act_shape[0])
        scaled_act = tensor * temp
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
        print(name)
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
num_samples = 512
seq_len = 512

for i in tqdm(range(num_samples)):
    input_ids = tokenizer(dataset_list[i]["text"], return_tensors="pt",
                            max_length=seq_len, truncation=True).input_ids.to(device)
    model(input_ids)

for h in hooks:
    h.remove()

# %%
print(act_scales)


