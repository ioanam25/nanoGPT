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
no_thinking = 1
thinking = 0
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

    train_indices = random.sample(range(400), 350)
    val_indices = [i for i in range(400) if i not in train_indices]

    train_sources = [subdirs[i] for i in train_indices]
    val_sources = [subdirs[i] for i in val_indices]

    return train_sources, val_sources

train_sources, val_sources = sources_split("/gpfs/data/oermannlab/users/im2178/lingua/sources")

print(val_sources)

print(train_sources[0])

prompts = []
responses = []
for t in train_sources:
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
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'source_0_no_thinking/1'# 'icl_no_thinking' # #'out-arc-char-8l_8h_02_3e-5new' # ignored if init_from is not 'resume'
index_prompt = 0
start = prompts[index_prompt] + '' # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
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

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt_10000.pt')
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


meta_path = 'data/arc/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(start)
print("true ans:")
print(responses[index_prompt])
print('---------------\n')
# run generation
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print('---------------')
samples_list = []
import csv

def write_to_csv(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Open the CSV file for writing (will overwrite if it already exists)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row: one start, one true_ans, and 10 sample columns
        header = ['start', 'true_ans'] + [f'sample_{i}' for i in range(10)]
        writer.writerow(header)
        
        # Write each row of data (start, true_ans, and 10 samples)
        for row in data:
            writer.writerow(row)

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
            print('---------------')
            samples_list.append(generated_output)

data = [[start, responses[index_prompt]] + samples_list]
write_to_csv('samples_csv/' + ckpt_path[:-3] + '.csv', data)
# import re

# def parse_matrix(input_str):
#     """
#     Parse the input matrix string into a list of lists (2D matrix).
#     """
#     # Remove outer square brackets and split the string by '][' to get individual rows
#     input_str = input_str.strip('[]')
#     rows = input_str.split('][')
    
#     # Convert each row into integers and store in a list of lists
#     matrix = [list(map(int, list(row))) for row in rows]
    
#     return matrix

# def is_matrix(matrix):
#     """
#     Check if the input is a valid matrix (list of lists) with rows of equal length.
#     """
#     if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
#         row_lengths = [len(row) for row in matrix]
#         if len(set(row_lengths)) == 1:  # All rows have the same length
#             return True
#     return False

# def compare_shapes(input_str, true_ans_str):
#     """
#     Compare the shapes of two matrix strings by parsing and checking their dimensions.
#     """
#     # Parse both input matrices
#     input_matrix = parse_matrix(input_str)
#     true_ans_matrix = parse_matrix(true_ans_str)
    
#     # Check if both are valid matrices
#     if not is_matrix(input_matrix):
#         return False, "Input matrix is not valid"
    
#     if not is_matrix(true_ans_matrix):
#         return False, "Reference matrix (true_ans) is not valid"
    
#     # Get shapes (number of rows and columns)
#     input_shape = (len(input_matrix), len(input_matrix[0]))
#     true_ans_shape = (len(true_ans_matrix), len(true_ans_matrix[0]))
    
#     # Compare the shapes
#     if input_shape == true_ans_shape:
#         return True
#     else:
#         return False

# # # Example input string and true_ans string
# # input_str = "[[7700000707007][7701002077007][6666666666666][6666666666666][7777700000707][7777702077700][7777702077070][7770702077007][0777770777077]]"
# # true_ans_str = "[[1][2][3]]"  # Example of another matrix in string form

# # # Compare the shapes of both matrices
# # result, message = compare_shapes(input_str, true_ans_str)
# # print(message)

# total = 0
# same_length = 0
# same_shape = 0
# results = []
# for p in tqdm(range(len(prompts))):
#     start = prompts[p]
#     if start.startswith('FILE:'):
#         with open(start[5:], 'r', encoding='utf-8') as f:
#             start = f.read()
#     start_ids = encode(start)
#     x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
#     true_ans = responses[p][:-1]
#     true_length = len(true_ans)
#     percent_common_chars = 0
#     prompt_too_long = (len(start) > 4096)
#     with torch.no_grad():
#         with ctx:
#             exact_curr = 0
#             same_length_curr = 0
#             max_common_chars = 0
#             same_shape_curr = 0
#             closest_ans = ''
#             exact_ans = ''
#             for k in range(num_samples):
#                 x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
#                 output_tokens = []  # To store generated tokens
#                 while True:
#                     y = model.generate(x, 1, temperature=temperature, top_k=top_k)  # Generate one token
#                     token = y[0, -1].item()  # Get the last generated token
#                     output_tokens.append(token)  # Append the token to the output list

#                     # Check if the token is 'F' (ensure you encode 'F' properly)
#                     if token == stoi['F']:  # or however you access the index for 'F'
#                         break

#                     if len(output_tokens) > 1000:
#                         break

#                     # Optionally update x with the new token to continue generation
#                     x = torch.cat([x, y[:, -1:]], dim=1)  # Append the new token to the input

#                 # Decode the generated tokens
#                 generated_output = decode(output_tokens)
#                 generated_ans = generated_output[:-1] # generated_output.rsplit('t', 1)[-1]

#                 if generated_ans == true_ans:
#                     exact_curr += 1
#                     same_shape_curr += 1  
#                     closest_ans = generated_ans
#                     percent_common_chars = 1
#                 elif len(generated_ans) == len(true_ans) and exact_curr == 0:
#                     print(true_ans, generated_ans)
#                     if compare_shapes(generated_ans, true_ans):
#                         same_shape_curr += 1  
#                         common_chars = sum(c1 == c2 for c1, c2 in zip(generated_ans, true_ans))
#                         if common_chars > max_common_chars:
#                             max_common_chars = common_chars
#                             closest_ans = generated_ans
#                             percent_common_chars = max_common_chars / len(true_ans)

#             if exact_curr > 0:
#                 total += 1
#                 results.append((start, true_ans, prompt_too_long, closest_ans, true_length, same_shape_curr > 0, percent_common_chars, 'found'))
#             elif same_shape_curr:
#                 same_shape += 1
#                 results.append((start, true_ans, prompt_too_long, closest_ans, true_length, same_shape_curr > 0, percent_common_chars, ''))
#             else:
#                 results.append((start, true_ans, prompt_too_long, closest_ans, true_length, same_shape_curr > 0, percent_common_chars, ''))

# print('number of prompts', len(prompts))
# print('exact sol found', total)
# print('same shape sol found', same_shape)

# csv_file = "results_icl_10k_valset.csv"
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # Write the header
#     writer.writerow(["input_prompt", "true_ans", "prompt_too_long", "closest_ans", "true length", "is same shape", "percent common char", "exact solution found"])
#     # Write the data rows
#     writer.writerows(results)