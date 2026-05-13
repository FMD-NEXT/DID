import time 
import torch
import argparse
from tqdm import tqdm
import abc

from run_sample_did import calculate_perplexity, batch_entropy
from load_model import load_model_local_cfg
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from catsample import sample_categorical


class Sampler(abc.ABC):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, device=torch.device('cuda')):
        super().__init__()
        self.model = model
        self.batch_dims = batch_dims
        self.device = device
        self.strategy = strategy
        self.strategy_para = strategy_para
        self.token_dim = token_dim

    @abc.abstractmethod
    def sample(self, steps):
        raise NotImplementedError


class DiffusionSampler(Sampler):
    def __init__(self, method, model, noise, batch_dims, token_dim, BOS, strategy, strategy_para=None, eps=1e-5, device=torch.device('cuda'), schedule='uniform'):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.noise = noise
        self.eps = eps
        self.method = method
        self.update_cnt = 0
        self.BOS = BOS
        self.token_dim = token_dim
        self.schedule = schedule 

    @torch.no_grad()
    def sample(self, steps, packed_x_tensor_init=None, seqlens_init=None):
        return self.direct_sample(steps, packed_x_tensor_init, seqlens_init)

    @torch.no_grad()
    def direct_sample(self, steps, packed_x_tensor_init=None, seqlens_init=None):
        self.model.eval()

        batchsize, seq_len = self.batch_dims # seq_len: model length
        max_len = seq_len + 50 # for generated tokens exceeding the model length 
        
        packed_x_tensor = packed_x_tensor_init.clone().to(self.device)
        seqlens = seqlens_init.clone().to(self.device)

        # cache
        changed_mask = torch.ones(batchsize, dtype=torch.bool).to(self.device)

        # unpacked tis
        unpacked_tis = torch.zeros((batchsize, max_len, self.token_dim + 1), dtype=torch.float64).to(self.device) # (B, max_len, V + 1)

        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        
        if self.schedule == 'cosine':
            timesteps = torch.cos(torch.pi / 2 * (1 - timesteps)) # dense -> sparse

        total_d1 = 0
        total_d2 = 0 

        for i in range(steps):
            t0 = time.perf_counter()

            t = timesteps[i]
            dt = timesteps[i] - timesteps[i+1] 
            update_rate = self.get_update_rate(t, steps, dt) if i < steps - 1 else 1 + 1e-3

            seqid = torch.repeat_interleave(torch.arange(batchsize).to(self.device), seqlens) # (packed, )
            unpack_mask = torch.arange(max_len, device=self.device)[None, :] < seqlens[:, None]  # (B, max_len)

            
            # score
            if changed_mask.any():
                packed_changed_mask = changed_mask[seqid] 
                packed_changed_x_tensor = packed_x_tensor[packed_changed_mask]
                changed_seqlens = seqlens[changed_mask]
                changed_seqlens_init = seqlens_init[changed_mask]

                # NFE
                unpacked_changed_mask = unpack_mask & changed_mask[:, None]  # (B, max_len)
                rows, cols = torch.where(unpacked_changed_mask)
                unpacked_tis[rows, cols, :-1] = self.model(packed_changed_x_tensor, changed_seqlens, changed_seqlens_init).double().exp() 
                packed_tis = unpacked_tis[unpack_mask]

            
            t1 = time.perf_counter()

            # probs
            probs = packed_tis * update_rate
            probs[..., -1]  = 1 - probs.sum(-1, keepdim=False)
            VOID = self.token_dim

            # sampling insertions
            insertions = sample_categorical(probs.to(torch.float64)).int()
            
            # do not update prefix
            seq_mask = torch.arange(seqlens.max(), device=self.device)[None, :] < seqlens[:, None]
            prompt_mask = torch.arange(seqlens.max(), device=self.device)[None, :] < seqlens_init[:, None] - 1
            packed_insertion_mask = prompt_mask[seq_mask]
            insertions[packed_insertion_mask] = VOID

            # update seqlens
            seqlens_old = seqlens.clone()
            seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

            # update x 
            inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
            packed_x_tensor = inserted_with_void[inserted_with_void != VOID]

            changed_mask = seqlens != seqlens_old

            t2 = time.perf_counter()

            total_d1 += t1 - t0 
            total_d2 += t2 - t1

        res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]
        return [_x[1:-1] for _x in res] , total_d1, total_d2 # remove bos, eos
    
    def get_update_rate(self, t, steps, dt):
        # dt = (1 - self.eps) / steps
        curr_sigma, next_sigma = self.noise(t)[0], self.noise(t - dt)[0]
        d_curr_sigma = self.noise(t)[1]
        if self.method == 'tweedie':
            update_rate = ((-next_sigma).exp() - (-curr_sigma).exp()) / (1 - (-curr_sigma).exp())
        elif self.method == 'euler':
            update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
        return update_rate
    

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    # parser.add_argument("--model_path", default="../output/2025.08.12/112518", type=str)
    parser.add_argument("--model_path", default="../output/2025.08.20/021412", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_num", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--method", type=str, default="euler") # euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p
    parser.add_argument("--strategy_para", type=float, default=0.8) # p for top_p, no use when direct 
    parser.add_argument("--schedule", type=str, default='uniform') # uniform, cosine
    parser.add_argument("--prompt_len", type=int, default=768)


    args = parser.parse_args()
    print(args)
    
    val_batch = torch.load('val_batch.pt')

    device = torch.device(args.device)
    model, noise, cfg = load_model_local_cfg(args.model_path, device)
    token_dim = model.config.tokens # + 1
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    
    if args.method == 'euler' or args.method == 'tweedie':
        BOS = tokenizer.encode(tokenizer.bos_token)[0] 
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, BOS, args.strategy, args.strategy_para, device=device, schedule=args.schedule)
    else:
        raise ValueError(f"Method {args.method} is not valid.")
    print(sampler)


    # warmup 
    i=0
    cond0 = val_batch[args.batch_size * i:args.batch_size * (i + 1), :args.prompt_len]
    seqlens_init = (cond0 > -1).sum(-1).int().to(args.device)
    packed_x_tensor_init = cond0.flatten().to(args.device)

    _sample, d1, d2 = sampler.sample(32, packed_x_tensor_init, seqlens_init)


    import time 
    torch.cuda.synchronize(device)
    start_time = time.time()

    sample = []
    total_d1 = 0
    total_d2 = 0
    for i in tqdm(range(args.total_num // args.batch_size)):

        cond0 = val_batch[args.batch_size * i:args.batch_size * (i + 1), :args.prompt_len]
        seqlens_init = (cond0 > -1).sum(-1).int().to(args.device)
        packed_x_tensor_init = cond0.flatten().to(args.device)

        _sample, d1, d2 = sampler.sample(args.steps, packed_x_tensor_init, seqlens_init)

        sample += _sample
        total_d1 += d1 
        total_d2 += d2 

    torch.cuda.synchronize(device)
    elapsed_time = time.time() - start_time

    text_samples = tokenizer.batch_decode(sample)

    for i, text in enumerate(text_samples):
        print(f"==================================== Sample {i} ====================================")
        print()
        print(text)
        print()

    print(f"elapsed time per sample: {elapsed_time / args.total_num:.3f}") 
    
    print(f"d1 per sample: {total_d1 / args.total_num:.3f}") 
    
    print(f"d2 per sample: {total_d2 / args.total_num:.3f}") 

    with torch.no_grad():
        eval_model = GPT2LMHeadModel.from_pretrained(cfg.gpt_dir).to(device).eval()

        perplexity_batch_size = args.batch_size
        from math import ceil
        batches = ceil(len(sample) / perplexity_batch_size)

        total_perplexity = 0
        total_entropy = 0
        total_len = 0
        for i in range(batches):
            s = text_samples[i * perplexity_batch_size:(i + 1) * perplexity_batch_size] 

            s = tokenizer(s)['input_ids']

            perplexity = calculate_perplexity(s, eval_model, tokenizer)
            
            s = sample[i * perplexity_batch_size:(i + 1) * perplexity_batch_size]
            
            entropys = batch_entropy(s, token_dim)

            total_perplexity += perplexity
            total_entropy += entropys
            total_len += sum([len(_s) + 2 for _s in s]) / len(s)
            print(f'{[len(_s) + 2 for _s in s] = }')
        total_perplexity /= batches
        total_entropy /= batches
        total_len /= batches
    
    print({'GPT2 perplexity': total_perplexity})
    print(f"Generative Entropy: {total_entropy:.3f}")
    print(f"Average Length: {total_len:.3f}")

    

if __name__=="__main__":
    main()