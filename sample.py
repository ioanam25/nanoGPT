"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import random
import itertools
import arckit
import numpy as np
from tqdm import tqdm
import csv
from pathlib import Path

random.seed(10)
print("BEGIN PROGRAM")
train_set_og, eval_set_og = arckit.load_data()
no_thinking = 0
thinking = 1
long_thinking = 0
def sources_split(root_dir: str):
    random.seed(10)
    subdirs = []
    for subdir in Path(root_dir).iterdir():
        if not subdir.is_dir():
            continue
        chunk_files = list(subdir.glob("*.chunk.*.jsonl"))
        if chunk_files:
            subdirs.append(str(subdir.name))

    train_indices = random.sample(range(400), 320)
    val_indices = [i for i in range(400) if i not in train_indices]

    train_sources = [subdirs[i] for i in train_indices]
    val_sources = [subdirs[i] for i in val_indices]

    return train_sources, val_sources

train_sources, val_sources = sources_split("/gpfs/data/oermannlab/users/im2178/nanoGPT/sources_10k")
print(train_sources[:5])

prompts = []
responses = []
for t in val_sources:
    task = train_set_og[t]
    all_ex = task.train + task.test
    prompt = 'B'
    # print(len(all_ex))
    for (i, io) in enumerate(all_ex):
        input, output = io[0], io[1]
        input_str = '[' + ''.join([f"[{''.join(map(str, row))}]" for row in input]) + ']'
        output_str = '[' + ''.join([f"[{''.join(map(str, row))}]" for row in output]) + ']'
        if i < len(all_ex) - 1:
            row = 'i' + input_str + 'o' + output_str
        else:
            # long_think = 't' * (int)(np.log(len(input_str) + len(output_str)))
            if no_thinking:
                row = 'E' + 'T' + 'i' + input_str + 'o'
            elif thinking:
                row = 'E' + 'T' + 't' * (int)(np.log(len(prompt))) + 'i' + input_str + 'o'
            elif long_thinking:
                row = 'E' + 'T' + 't' * (len(prompt) // 3) + 'i' + input_str + 'o'
        prompt += row
          
    prompts.append(prompt)
    responses.append(output_str + 'F')

# -----------------------------------------------------------------------------
num_samples = 50 # number of samples to draw
max_new_tokens = 5000 # number of tokens generated in each sample
temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def write_to_csv(filename, start, responses, index_prompt, samples_list):
    data = [[start, responses[index_prompt]] + samples_list]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Open the CSV file for writing (will overwrite if it already exists)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row: one start, one true_ans, and len samples sample columns
        header = ['start', 'true_ans'] + [f'sample_{i}' for i in range(len(samples_list))]
        writer.writerow(header)
        
        # Write each row of data (start, true_ans, and 10 samples)
        for row in data:
            writer.writerow(row)


meta_path = 'data/arc/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

if no_thinking:
    # out_dirs = ['no_thinking/five_sources/size3/2/ckpt_40000.pt']
    # out_dirs = ['no_thinking/five_sources/size2/1/ckpt_19000.pt']
    # out_dirs = ['no_thinking/source_0/size1/3/ckpt_15000.pt', 'no_thinking/source_0/size2/1/ckpt_21000.pt', 
    #             'no_thinking/source_1/size1/3/ckpt_23000.pt', 'no_thinking/source_1/size2/1/ckpt_29000.pt', 
    #             'no_thinking/source_2/size1/3/ckpt_24000.pt', 'no_thinking/source_2/size2/2/ckpt_12000.pt', 
    #             'no_thinking/source_3/size1/5/ckpt_10000.pt', 'no_thinking/source_3/size2/2/ckpt_16000.pt', 
    #             'no_thinking/source_4/size1/2/ckpt_29000.pt', 'no_thinking/source_4/size2/5/ckpt_20000.pt']

    out_dirs = [# 'no_thinking/ten_sources/size3/1/ckpt_43000.pt', 
                # 'no_thinking/twenty_sources/size3/1/ckpt_85000.pt', # 2
                # 'no_thinking/forty_sources/size3/1/ckpt_66000.pt', # 2
                # 'no_thinking/eighty_sources/size3/1/ckpt_6000.pt',
                # 'no_thinking/onesixty_sources/size3/1/ckpt_8000.pt',
                'no_thinking/threetwenty_sources/size3/1/ckpt_63000.pt' # running 3
                ]


elif thinking:
    # out_dirs = ['thinking/five_sources/size3/2/ckpt_13000.pt']
    # out_dirs = ['thinking/five_sources/size2/2/ckpt_19000.pt']
    # out_dirs = ['thinking/source_0/size1/3/ckpt_49000.pt', 'thinking/source_0/size2/4/ckpt_22000.pt', 
    #             'thinking/source_1/size1/5/ckpt_16000.pt', 'thinking/source_1/size2/5/ckpt_34000.pt', 
    #             'thinking/source_2/size1/4/ckpt_29000.pt', 'thinking/source_2/size2/3/ckpt_18000.pt', 
    #             'thinking/source_3/size1/5/ckpt_24000.pt', 'thinking/source_3/size2/3/ckpt_13000.pt', 
    #             'thinking/source_4/size1/2/ckpt_48000.pt', 'thinking/source_4/size2/5/ckpt_18000.pt']

    out_dirs = [# ' thinking/ten_sources/size3/1/ckpt_23000.pt',
                #'thinking/twenty_sources/size3/1/ckpt_33000.pt',
                # 'thinking/forty_sources/size3/1/ckpt_8000.pt',
                # 'thinking/eighty_sources/size3/1/ckpt_13000.pt',
                # 'thinking/onesixty_sources/size3/1/ckpt_9000.pt',
                'thinking/threetwenty_sources/size3/1/ckpt_20000.pt' # running 2
                ]

for dir in out_dirs:
    # model
    print(dir)
    checkpoint = torch.load(dir, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # source
    # index_prompt = (int)(dir[dir.find('/', dir.find('/') + 1) - 1])
    for index_prompt in range(80):
        samples_list = []
        start = prompts[index_prompt]
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        print('prompt: ', start)
        print('true ans: ', responses[index_prompt])

        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                    output_tokens = []  # To store generated tokens
                    while True:
                        y = model.generate(x, 1, temperature=temperature, top_k=top_k)  # Generate one token
                        token = y[0, -1].item()  # Get the last generated token
                        output_tokens.append(token)  # Append the token to the output list

                        # Check if the token is 'F' (ensure you encode 'F' properly)
                        if token == stoi['F']:  # or however you access the index for 'F'
                            break

                        if len(output_tokens) > 4096:
                            break

                        # Optionally update x with the new token to continue generation
                        x = torch.cat([x, y[:, -1:]], dim=1)  # Append the new token to the input

                    # Decode the generated tokens
                    generated_output = decode(output_tokens)
                    print(generated_output)
                    samples_list.append(generated_output)
                    if generated_output == responses[index_prompt]:
                        print("FOUND")
                        break

        write_to_csv('samples_csv/' + str(index_prompt) + '/' + dir[:-3] + '.csv', start, responses, index_prompt, samples_list)
