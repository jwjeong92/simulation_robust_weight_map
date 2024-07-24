from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

model_name = 'facebook/opt-6.7b'
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

dataset = dataset.shuffle(seed=42)
dataset_list = []
for ii in range(len(dataset)):
    if dataset[ii]['text'] != '':
        dataset_list.append(dataset[ii])

num_samples = 4
seq_len = 64

for i in tqdm(range(num_samples)):
    input_ids = tokenizer(dataset_list[i]["text"], return_tensors="pt",
                            max_length=seq_len, truncation=True).input_ids.to(device)
    model(input_ids)

