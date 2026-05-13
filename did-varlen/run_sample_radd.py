import torch
import argparse

from load_model import load_model, load_model_radd
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sampling_radd import DiffusionSampler
from torch.cuda.nvtx import range_push, range_pop, mark 
from tqdm import tqdm 

import torch.nn.functional as F 
from copy import deepcopy
def calculate_perplexity(batch, model, tokenizer):
    max_length = 1024#max(len(seq) for seq in batch)
    batch = deepcopy(batch)
    batch = [seq[:max_length] for seq in batch] 
    # print(max_length)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    padded_batch = [seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in batch if seq]
    # print(padded_batch)
    input_ids = torch.tensor(padded_batch).to(model.device)
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits

    logits = logits.transpose(-1, -2)  #  [batch, vocab, seq]
    
    loss = F.cross_entropy(
        logits[..., :-1], 
        input_ids[..., 1:], 
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    
    valid_tokens = (input_ids[..., 1:] != tokenizer.pad_token_id).sum(dim=-1)
    
    seq_loss = loss.sum(dim=-1) / valid_tokens
    # return seq_loss[~seq_loss.isnan()].mean()
    
    batch_perplexity = seq_loss.exp()
    print(seq_loss.exp())

    thresh = 300
    print((batch_perplexity < thresh).float().mean())
    
    return batch_perplexity[batch_perplexity < thresh].mean()#.item()
    
def batch_entropy(token_ids, vocab_size):
    ents = []
    for x in token_ids:
        try:
            x = torch.tensor(x)
            counts = torch.nn.functional.one_hot(x, vocab_size).sum(dim=0)
            probs = counts / x.size(0)
            nz = probs > 0
            ent =  -torch.sum(probs * torch.log2(probs.where(nz, torch.ones_like(probs))))
            ents.append(ent)
        except: pass 
    return torch.tensor(ents).mean()


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    # parser.add_argument("--model_path", default="output/2025.09.11/091943", type=str) # 1024
    
    parser.add_argument("--model_path", default="output/2025.09.12/121149", type=str) # 512
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_num", type=int, default=1)
    parser.add_argument("--length", type=int, default=512) # 
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--method", type=str, default="euler") # ordered, euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k
    parser.add_argument("--strategy_para", type=float, default=0.8) # p for top_p, k for top_k, no use when direct 


    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    
    device = torch.device(args.device)
    model, noise = load_model_radd(args.model_path, device)
    token_dim = model.config.tokens + 1
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #GPT2TokenizerFast.from_pretrained('gpt2')
    BOS = tokenizer.cls_token_id
    EOS = tokenizer.sep_token_id

    if args.method == 'euler' or args.method == 'tweedie':
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, BOS, args.strategy, args.strategy_para, device=device)
    else:
        raise ValueError(f"Method {args.method} is not valid.")
    print(sampler)


    # warmup 
    sampler.sample(10)
    
    import time 
    # start_event = torch.cuda.Event(enable_timing=True)
    # stop_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    # start_event.record()
    start_time = time.time()

    sample = []
    total_d1 = 0
    total_d2 = 0
    for text in tqdm(range(args.total_num // args.batch_size)):
        range_push(f"sampling {text}")
        _sample, d1, d2 = sampler.sample(args.steps)
        sample += _sample
        total_d1 += d1 
        total_d2 += d2 
        range_pop() 

    torch.cuda.synchronize(device)
    # stop_event.record()
    # torch.cuda.synchronize(device)
    
    # elasped_time = start_event.elapsed_time(stop_event)
    elapsed_time = time.time() - start_time

    text_samples = tokenizer.batch_decode(sample)

    for i, text in enumerate(text_samples):
        print(f"==================================== Sample {i} ====================================")
        print(text)
        
    print(f"elapsed time per sample: {elapsed_time / args.total_num:.3f}") 
    
    print(f"d1 per sample: {total_d1 / args.total_num:.3f}") 
    
    print(f"d2 per sample: {total_d2 / args.total_num:.3f}") 


    # rm cls, pad 
    for i in range(len(sample)):
        sample[i] = [tok for tok in sample[i] if tok not in [tokenizer.pad_token_id, BOS]]
        # print(i, sample[i])

    # exit(0)
    text_samples = tokenizer.batch_decode(sample)

    with torch.no_grad():
        # eval_model = GPT2LMHeadModel.from_pretrained(cfg.gpt_dir).to(device).eval()
        from transformers import AutoModelForCausalLM 
        # model_name = 'Llama-3___2-3B'
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eval_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        eval_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

        perplexity_batch_size = args.batch_size
        from math import ceil
        batches = ceil(len(sample) / perplexity_batch_size)

        total_perplexity = 0
        total_entropy = 0
        total_len = 0
        lengths = []
        for i in range(batches):
            s = text_samples[i * perplexity_batch_size:(i + 1) * perplexity_batch_size] 
            s = tokenizer(s)['input_ids']

            perplexity = calculate_perplexity(s, eval_model, tokenizer)
            entropys = batch_entropy(s, tokenizer.vocab_size + 1)

            total_perplexity += perplexity
            total_entropy += entropys
            total_len += sum([len(_s)  for _s in s]) / len(s)
            lengths += [len(_s) for _s in s]
            
        total_perplexity /= batches
        total_entropy /= batches
        total_len /= batches
    
    print({'GPT2 PPL': total_perplexity})
    print(f"Generative Entropy: {total_entropy:.3f}")
    print(f"Average Length: {total_len:.3f}")


    # import json 
    # file_ = f'{args.model_path}/lengths_mr_steps_{args.steps}'
    # with open(file_, "w") as f:
    #     f.write(
    #         json.dumps(
    #             {
    #                 "lengths": lengths #.tolist()
    #             }
    #         )
    #     )

if __name__=="__main__":
    main()