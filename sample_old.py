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

random.seed(10)
train_set_og, eval_set_og = arckit.load_data()
random_indices_train = random.sample(range(400), 350)
not_in_random_indices_train = [i for i in range(400) if i not in random_indices_train]
random_indices_eval = random.sample(range(400), 350)
not_in_random_indices_eval = [i for i in range(400) if i not in random_indices_eval]
train_set = [train_set_og[i] for i in random_indices_train] + [eval_set_og[i] for i in random_indices_eval]
test_set = [train_set_og[i] for i in not_in_random_indices_train] + [eval_set_og[i] for i in not_in_random_indices_eval]

prompts = []
responses = []
for t in range(len(test_set)):
    task = test_set[t]
    all_ex = task.train + task.test
    combinations = list(itertools.combinations(all_ex, 3))
    for combination in combinations:
        permutations = list(itertools.permutations(combination))
        for perm in permutations:
            prompt = 'B'
            for (i, io) in enumerate(perm):
                input, output = io[0], io[1]
                input_str = '[' + ''.join([f"[{''.join(map(str, row))}]" for row in input]) + ']'
                output_str = '[' + ''.join([f"[{''.join(map(str, row))}]" for row in output]) + ']'
                if i < 2:
                    row = 'i' + input_str + 't' + 'o' + output_str
                if i == 2:
                    long_think = 't' * (int)(np.log(len(input_str) + len(output_str)))
                    row = 'E' + long_think + 'T' + 'i' + input_str + 'o'
                prompt += row
          
            prompts.append(prompt)
            responses.append(output_str + 'F')

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-arc-char-8l_8h_02_3e-5new' # ignored if init_from is not 'resume'
index_prompt = 1000
start = prompts[index_prompt] + '' # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 5000 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
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

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

'''# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)'''

meta_path = 'data/arc/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# print(start)
# print("true ans:")
# print(responses[index_prompt])
# print('---------------\n')
# # run generation
# # with torch.no_grad():
# #     with ctx:
# #         for k in range(num_samples):
# #             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
# #             print(decode(y[0].tolist()))
# #             print('---------------')
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
#             output_tokens = []  # To store generated tokens
#             while True:
#                 y = model.generate(x, 1, temperature=temperature, top_k=top_k)  # Generate one token
#                 token = y[0, -1].item()  # Get the last generated token
#                 output_tokens.append(token)  # Append the token to the output list

#                 # Check if the token is 'F' (ensure you encode 'F' properly)
#                 if token == stoi['F']:  # or however you access the index for 'F'
#                     break

#                 if len(output_tokens) > 1000:
#                     break

#                 # Optionally update x with the new token to continue generation
#                 x = torch.cat([x, y[:, -1:]], dim=1)  # Append the new token to the input

#             # Decode the generated tokens
#             generated_output = decode(output_tokens)
#             print(generated_output)
#             print('---------------')

total = 0
same_length = 0
results = []
for p in tqdm(range(len(prompts))):
    start = prompts[p]
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    true_ans = responses[p]
    true_length = len(true_ans)
    percent_common_chars = 0
    with torch.no_grad():
        with ctx:
            exact_curr = 0
            same_length_curr = 0
            max_common_chars = 0
            closest_ans = ''
            exact_ans = ''
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

                    if len(output_tokens) > 1000:
                        break

                    # Optionally update x with the new token to continue generation
                    x = torch.cat([x, y[:, -1:]], dim=1)  # Append the new token to the input

                # Decode the generated tokens
                generated_output = decode(output_tokens)
                generated_ans = generated_output.rsplit('t', 1)[-1]

                if generated_ans == true_ans:
                    exact_curr += 1
                    closest_ans = generated_ans
                    percent_common_chars = 1
                elif len(generated_ans) == len(true_ans) and exact_curr == 0:
                    same_length_curr += 1
                    common_chars = sum(c1 == c2 for c1, c2 in zip(generated_ans, true_ans))
                    if common_chars > max_common_chars:
                        max_common_chars = common_chars
                        closest_ans = generated_ans
                        percent_common_chars = max_common_chars / len(true_ans)  

            if exact_curr > 0:
                total += 1
                results.append((start, true_ans, closest_ans, true_length, percent_common_chars, 'found'))
            elif same_length_curr > 0:
                same_length += 1
                results.append((start, true_ans, closest_ans, true_length, percent_common_chars, ''))
            else:
                results.append((start, true_ans, closest_ans, true_length, percent_common_chars, ''))

print('number of prompts', len(prompts))
print('exact sol found', total)
print('same length sol found', same_length)

csv_file = "results_new.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["input_prompt", "true_ans", "closest_ans", "true length", "percent common char", "exact solution found"])
    # Write the data rows
    writer.writerows(results)