import torch
import argparse

from load_model import load_model, load_model_radd
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sampling_radd import OrderedSampler,DiffusionSampler
from torch.cuda.nvtx import range_push, range_pop, mark 
from tqdm import tqdm 

import torch.nn.functional as F 
def calculate_perplexity(batch, model, tokenizer):
    
    max_length = max(len(seq) for seq in batch)
    # print(max_length)
    
    # padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    padded_batch = [seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in batch]
    # print(padded_batch)
    input_ids = torch.tensor(padded_batch).to(model.device)
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
    
    logits = logits.transpose(-1, -2)  # [batch, vocab, seq]
    
    loss = F.cross_entropy(
        logits[..., :-1],  
        input_ids[..., 1:], 
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    # print(loss)
    valid_tokens = (input_ids[..., 1:] != tokenizer.pad_token_id).sum(dim=-1)
    
    seq_loss = loss.sum(dim=-1) / valid_tokens
    
    batch_perplexity = seq_loss.exp().mean()
    
    return batch_perplexity
    
def batch_entropy(token_ids, vocab_size):
    ents = []
    for x in token_ids:
        x = torch.tensor(x)
        counts = torch.nn.functional.one_hot(x, vocab_size).sum(dim=0)
        probs = counts / x.size(0)
        nz = probs > 0
        ent =  -torch.sum(probs * torch.log2(probs.where(nz, torch.ones_like(probs))))
        ents.append(ent)
    return torch.tensor(ents).mean()


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
    parser.add_argument("--schedule", type=str, default='unif') # unif, d2s, s2d


    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    
    device = torch.device(args.device)
    model, noise = load_model_radd(args.model_path, device)
    token_dim = model.config.tokens + 1
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    order =  torch.arange(0,1024)
    if args.method == 'ordered':
        sampler = OrderedSampler(model, (args.batch_size, args.length), token_dim, args.strategy, args.strategy_para, order, device=device)
    elif args.method == 'euler' or args.method == 'tweedie':
        BOS = tokenizer.encode(tokenizer.bos_token)[0] 
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, BOS, args.strategy, args.strategy_para, device=device, schedule=args.schedule)
    else:
        raise ValueError(f"Method {args.method} is not valid.")


    # warmup 
    sampler.sample(args.steps)
    
    import time 

    torch.cuda.synchronize(device)
    
    start_time = time.time()

    sample = []
    total_d1 = 0
    total_d2 = 0
    for _ in tqdm(range(args.total_num // args.batch_size)):
        
        _sample, d1, d2 = sampler.sample(args.steps)
        sample += _sample
        total_d1 += d1 
        total_d2 += d2 
        range_pop() 

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