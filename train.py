"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random
import json
from model import GPTConfig, GPT
from pathlib import Path

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
curr_source = 4
no_thinking = 0
thinking = 0
long_thinking = 0
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
train_batch_index = 0

# tokenizer
meta_path = 'data/arc/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# poor man's data loader
data_dir = os.path.join('data', dataset)
import random
import numpy as np
import torch
from typing import Dict, Any, Iterator, Optional, TypedDict

def sources_split(root_dir: str):
    random.seed(10)
    subdirs = []
    for subdir in Path(root_dir).iterdir():
        if not subdir.is_dir():
            continue
        chunk_files = list(subdir.glob("*.chunk.*.jsonl"))
        if chunk_files:
            subdirs.append(str(chunk_files[0]))

    train_indices = random.sample(range(400), 350)
    val_indices = [i for i in range(400) if i not in train_indices]

    train_sources = [subdirs[i] for i in train_indices]
    val_sources = [subdirs[i] for i in val_indices]
    random.seed(None)

    return train_sources, val_sources

train_sources, val_sources = sources_split("/gpfs/data/oermannlab/users/im2178/nanoGPT/sources_10k")
# print(train_sources, val_sources)

def read_jsonl_block(file_path: str):
    """Read all lines from a JSONL file into memory."""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

def sample_and_combine_lines(lines: list, rng: np.random.Generator) -> dict:
    """Sample 3 random lines and combine their text/content."""
    random_num = rng.integers(0, 800)
    if random_num < 160:
        num_lines = 4
    elif random_num < 160 + 450:
        num_lines = 3
    elif random_num < 160 + 450 + 40:
        num_lines = 5
    else:
        num_lines = 2
    sampled_lines = rng.choice(lines, size=num_lines, replace=False)
    combined_text = "B" # B is begin training symbol
    for line in sampled_lines:
        content_key = "text" if "text" in line else "content"
        combined_text += line[content_key] 
    return combined_text + "ET" # E is end training, T is begin testing


def get_batch(split):
    # print("in get batch")
    if split == 'train':
        sources = train_sources # random.choice(train_sources)
        source = sources[curr_source]
        lines = read_jsonl_block(source)[:9000]
    else:
        sources = train_sources # random.choice(val_sources)
        source = sources[curr_source]
        lines = read_jsonl_block(source)[9000:]
    # source = sources[0] # random.choice(sources)
    # print("SOURCE", source)
    # lines = read_jsonl_block(source)
    rng = np.random.default_rng()
    ans_lines = rng.choice(lines, size=batch_size, replace=False) #random.randint(1000, (batch_size, ))
    x = []
    y = []
    # print("we have ans lines", len(ans_lines))
    # print(block_size)
    for line in ans_lines: # we fill batch
        full_task = 'P' * (block_size + 1)
        while len(full_task) > block_size:
            # print("enter while 1")
            prompt_lines = sample_and_combine_lines(lines, rng) 
            if no_thinking:
                full_task = prompt_lines + line["text"] + 'F'
            elif thinking:
                full_task = prompt_lines + 't' * (int)(np.log(len(prompt_lines))) + line["text"] + 'F'
            elif long_thinking:
                full_task = prompt_lines + 't' * (len(prompt_lines) // 3) + line["text"] + 'F'
            # print(len(full_task))

        # print("exit while 1")
        # print("full task before padding", full_task)
        while len(full_task) < block_size:
            # print("enter while 2")
            full_task += 'P'
        # print(print("exit while 2"))
        x.append(torch.tensor(encode(full_task))) # x = ....ansPPPP y = PPPPansPPP
        # print(line['text'])
        if no_thinking:
            target = 'P' * (len(prompt_lines) - 1 ) + line['text'] + 'F'
        elif thinking:
            target = 'P' * (len(prompt_lines) - 1 + (int)(np.log(len(prompt_lines)))) + line['text'] + 'F'
        elif long_thinking:
            target = 'P' * (len(prompt_lines) - 1 + (len(prompt_lines) // 3)) + line['text'] + 'F'
        # print("target before padding", target)
        while len(target) < block_size:
            target += 'P'
        y.append(torch.tensor(encode(target)))
    
    # print("time to stack")
    x = torch.stack(x)
    y = torch.stack(y)
    # print(x.shape, y.shape)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0 # 0
best_val_loss = 1e9 #1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_39000.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # print("eval iters", eval_iters)
        for k in range(eval_iters):
            # print(k)
            X, Y = get_batch(split)
            # print('batch', X, Y)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

early_stopping_patience = 10  # Number of steps to wait for improvement
early_stopping_min_delta = 1e-3  # Minimum change in loss to count as an improvement
early_stopping_counter = 0  # Counter to track how many steps since last improvement

# training loop
X, Y = get_batch('train') # fetch the very first batch
print("batch shapes", X.shape, Y.shape)
print('batch', X, Y)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
train_losses = []
val_losses = []
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # print("in eval")
        losses = estimate_loss()
        # print("estimated loss")
        # print(losses)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        print(f"Iter: {iter_num}, "
          f"Train Loss: {losses['train']:.4f}, "
          f"Val Loss: {losses['val']:.4f}, "
          f"Learning Rate: {lr:.6e}, "
          f"MFU: {running_mfu * 100:.2f}%")
        
        train_loss = losses['train']
        val_loss = losses['val']
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0 

            if val_loss < best_val_loss or always_save_checkpoint:
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt_' + str(iter_num) + '.pt'))
                    print(best_val_loss, iter_num)
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at iteration {iter_num} due to no improvement in validation loss.")
            np.save(out_dir + '/train_losses.npy', train_losses)
            np.save(out_dir + '/val_losses.npy', val_losses)
            break
    if iter_num == 0 and eval_only:
        np.save(out_dir + '/train_losses.npy', train_losses)
        np.save(out_dir + '/val_losses.npy', val_losses)
        break
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # print('in training')
        train_batch_index += batch_size  # Increment the batch counter
        # print(train_batch_index/batch_size)
        X, Y = get_batch('train')
        # print('batch', X, Y)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
