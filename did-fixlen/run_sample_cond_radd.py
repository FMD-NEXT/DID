import time 
import abc
from tqdm import tqdm 
import torch
import argparse

from load_model import load_model_radd
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from run_sample_radd import calculate_perplexity, batch_entropy
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
    def __init__(self, method, model, noise, batch_dims, token_dim, BOS, strategy, strategy_para=None, eps=1e-5, device=torch.device('cuda'), schedule='unif'):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.noise = noise
        self.eps = eps
        self.method = method
        self.BOS = BOS
        self.update_cnt = 0
        self.schedule = schedule 

    @torch.no_grad()
    def sample(self, steps,cond0, prompt_len):
        return self.direct_sample(steps, cond0, prompt_len)

    @torch.no_grad()
    def direct_sample(self, steps, cond0, prompt_len):
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x[:, :prompt_len] = cond0 
        
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        if self.schedule == 'cosine':
            timesteps = torch.cos(torch.pi / 2 * (1 - timesteps)) # dense -> sparse

        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        p_condition = torch.zeros(*self.batch_dims, self.token_dim, dtype=torch.float64).to(self.device)

        mask_num_log = []
        total_d1 = 0
        total_d2 = 0 
        for i in range(steps):
            
            t0 = time.perf_counter()

            t = timesteps[i]
            dt = timesteps[i] - timesteps[i+1] 
            update_rate = self.get_update_rate(t, steps, dt) if i < steps - 1 else 1 + 1e-3
            mask = x == self.token_dim - 1
            mask_num_log.append(mask.sum().item() / self.batch_dims[0])

            if changed.any():
                p_condition[changed] = self.model(x[changed]).double().exp()
                p_condition_mask = p_condition[mask]
            
            t1 = time.perf_counter()
            
            probs_mask = p_condition_mask * update_rate
            probs_mask[..., -1] = 1 - update_rate
            
            update_x_mask = sample_categorical(probs_mask.to(torch.float64))
            
            x_old = x.clone()
            x[mask] = update_x_mask
            changed = (x != x_old).any(dim=-1)
            self.update_cnt += changed.sum().item()
            
            t2 = time.perf_counter()
            total_d1 += t1 - t0 
            total_d2 += t2 - t1

        x = x[:, 1:].tolist() 
        return x, total_d1, total_d2

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
    parser.add_argument("--model_path", default="radd-lambda-dce-medium", type=str)
    # parser.add_argument("--model_path", default="radd-lambda-dce", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_num", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", type=str, default="tweedie") # ordered, euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k
    parser.add_argument("--strategy_para", type=float, default=0.8) # p for top_p, k for top_k, no use when direct 
    parser.add_argument("--schedule", type=str, default='uniform') # uniform, cosine
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    val_batch = torch.load('val_batch.pt')
    
    device = torch.device(args.device)
    model, noise = load_model_radd(args.model_path, device)
    token_dim = model.config.tokens + 1
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    
    if args.method == 'euler' or args.method == 'tweedie':
        BOS = tokenizer.encode(tokenizer.bos_token)[0] 
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, BOS, args.strategy, args.strategy_para, device=device, schedule=args.schedule)
    else:
        raise ValueError(f"Method {args.method} is not valid.")


    # warmup 
    i=0
    cond0 = val_batch[args.batch_size * i:args.batch_size * (i + 1), :args.prompt_len]
    _sample, d1, d2 = sampler.sample(32, cond0, args.prompt_len)

    

    torch.cuda.synchronize(device)
    start_time = time.time()

    sample = []
    total_d1 = 0
    total_d2 = 0
    for text in tqdm(range(args.total_num // args.batch_size)):
        
        cond0 = val_batch[args.batch_size * i:args.batch_size * (i + 1), :args.prompt_len]
        _sample, d1, d2 = sampler.sample(args.steps, cond0, args.prompt_len)

        sample += _sample
        total_d1 += d1 
        total_d2 += d2 

    torch.cuda.synchronize(device)
    elapsed_time = time.time() - start_time

    text_samples = tokenizer.batch_decode(sample)

    for i, text in enumerate(text_samples):
        print(f"{i}=================================================")
        print(text)


    print(f"elapsed time per sample: {elapsed_time / args.total_num:.3f}") 
    
    print(f"d1 per sample: {total_d1 / args.total_num:.3f}") 
    
    print(f"d2 per sample: {total_d2 / args.total_num:.3f}") 

    with torch.no_grad():
        
        eval_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()

        from math import ceil
        perplexity_batch_size = args.batch_size
        batches = ceil(len(sample) / perplexity_batch_size)

        total_perplexity = 0
        total_entropy = 0
        total_len = 0
        for i in range(batches):
            s = sample[i * perplexity_batch_size:(i + 1) * perplexity_batch_size] 
            perplexity = calculate_perplexity(s, eval_model, tokenizer)

            s = sample[i * perplexity_batch_size:(i + 1) * perplexity_batch_size]
            entropys = batch_entropy(s, token_dim)

            total_perplexity += perplexity
            total_entropy += entropys.mean().item()
            total_len += sum([len(_s)  for _s in s]) / len(s)
            # print(f'{[len(_s)  for _s in s] = }')
        total_perplexity /= batches
        total_entropy /= batches
        total_len /= batches
    
    print({'GPT2 perplexity': total_perplexity})
    print(f"Generative Entropy: {total_entropy:.3f}")
    print(f"Average Length: {total_len:.3f}")

    


if __name__=="__main__":
    main()