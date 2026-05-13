import torch
import argparse

from load_model import load_model_local_cfg
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sampling import DiffusionSampler
from torch.cuda.nvtx import range_push, range_pop, mark 
from tqdm import tqdm

import torch.nn.functional as F 


def batch_entropy(token_ids, vocab_size):
    ents = []
    for x in token_ids:
        x = torch.tensor(x)
        counts = torch.nn.functional.one_hot(x, vocab_size).sum(dim=0)
        probs = counts / x.size(0)
        nz = probs > 0
        ent = -torch.sum(probs * torch.log2(probs.where(nz, torch.ones_like(probs))))
        ents.append(ent)
    return torch.tensor(ents).mean()

from copy import deepcopy
def calculate_perplexity(batch, model, tokenizer):
    max_length = 1024#max(len(seq) for seq in batch)
    batch = deepcopy(batch)
    batch = [seq[:max_length] for seq in batch] 
    # print(max_length)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # print(batch)
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
    # return seq_loss.mean()
    
    batch_perplexity = seq_loss.exp().mean()
    # batch_perplexity = seq_loss.mean()
    # print(seq_loss.exp())
    
    return batch_perplexity#.item()

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--dataset", type=str, default="st") # st / lm 
    # parser.add_argument("--model_path", default="./output/2025.09.03/113744", type=str) # stories  python run_sample.py --dataset st --steps 256 --total_num 1024 --batch_size 32 --strategy top_p --strategy_para 0.9 > ds_st.txt
    parser.add_argument("--model_path", default="./output/2025.09.04/090057", type=str) # lm1b python run_sample.py --dataset lm --steps 32 --total_num 1024 --batch_size 32 --strategy top_p --strategy_para 0.9 > ds_lm.txt
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_num", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--method", type=str, default="euler") # ordered, euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k
    parser.add_argument("--strategy_para", type=float, default=0.9) # p for top_p, k for top_k, no use when direct 



    args = parser.parse_args()
    print(args)

    # if args.dataset == 'st':
        
    #     # args.model_path = "./output/2025.09.03/113744"
    #     args.model_path = "./output/2025.09.12/162651"
    #     args.length = 1024
    # elif args.dataset == 'lm':
    #     args.model_path = "./output/2025.09.04/090057"
    #     args.length = 128
    
    device = torch.device(args.device)
    model, noise, cfg = load_model_local_cfg(args.model_path, device)
    token_dim = model.config.tokens # + 1

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #GPT2TokenizerFast.from_pretrained('gpt2')
    BOS = tokenizer.cls_token_id
    EOS = tokenizer.sep_token_id

    order =  torch.arange(0,1024)
    if args.method == 'ordered':
        sampler = OrderedSampler(model, (args.batch_size, args.length), token_dim, args.strategy, args.strategy_para, order, device=device)
    elif args.method == 'euler' or args.method == 'tweedie':
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, BOS, args.strategy, args.strategy_para, device=device)
    else:
        raise ValueError(f"Method {args.method} is not valid.")
    print(sampler)


    # warmup 
    range_push("warmup1")
    sampler.sample(32)
    range_pop() 


    import time 
    # start_event = torch.cuda.Event(enable_timing=True)
    # stop_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    # start_event.record()
    start_time = time.time()

    sample = []
    total_d1 = 0
    total_d2 = 0
    for i in tqdm(range(args.total_num // args.batch_size)):
        range_push(f"sampling {i}")
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
        print()
        print(text)
        print()
        
    print(f"elapsed time per sample: {elapsed_time / args.total_num:.3f}") 
    
    print(f"d1 per sample: {total_d1 / args.total_num:.3f}") 
    
    print(f"d2 per sample: {total_d2 / args.total_num:.3f}") 


    with torch.no_grad():
        from transformers import AutoModelForCausalLM, AutoTokenizer 
        # model_name = 'Llama-3___2-3B'
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eval_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token

        eval_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

        # if isinstance(sample, torch.Tensor):
        #     batches = sample.shape[0] // cfg.eval.perplexity_batch_size
        # elif isinstance(sample, list):
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
            total_len += sum([len(_s) for _s in s]) / len(s)
            lengths += [len(_s) for _s in s]
            
        total_perplexity /= batches
        total_entropy /= batches
        total_len /= batches
    
    print({'GPT2 PPL': total_perplexity})
    print(f"Generative Entropy: {total_entropy:.3f}")
    print(f"Average Length: {total_len:.3f}")

    # import json 
    # file_ = f'{args.model_path}/lengths_mr_steps_{args.steps}_{args.strategy}'
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