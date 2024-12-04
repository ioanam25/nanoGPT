# train a miniature character-level arc model
# good for debugging and playing on macbooks and such

out_dir = 'source_0_no_thinking/1'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 1000 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'arc'
wandb_run_name = 'icl'

dataset = 'arc'
gradient_accumulation_steps = 8

batch_size = 8
block_size = 4096 # 8192 #4096 # context of up to 256 previous characters

# baby GPT model :)
# n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

n_layer = 15 # 30 ~ 800M  n_layer=48, n_head=25, n_embd=1600
n_head = 12 # 24 ~ 800M 12
n_embd = 768 * 2 # 768
dropout = 0

learning_rate = 3e-5 # 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000 # prev 5000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 3e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

