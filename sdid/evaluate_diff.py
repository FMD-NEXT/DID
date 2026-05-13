'''
This file is inspired by the code provided by the author of https://arxiv.org/abs/2406.11473
'''
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer
from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file

from datetime import timedelta
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)

# from eval.math_normalization import normalize_final_answer

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remap_weights(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'mlp.swiglu.w1.weight' in key:
            new_key = key.replace('mlp.swiglu.w1.weight', 'mlp.fc_1.weight')
        elif 'mlp.swiglu.w2.weight' in key:
            new_key = key.replace('mlp.swiglu.w2.weight', 'mlp.fc_2.weight')
        elif 'mlp.swiglu.w3.weight' in key:
            new_key = key.replace('mlp.swiglu.w3.weight', 'mlp.proj.weight')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

@register_model("mdlm")
class MDLMEvalHarness(LM):
    def __init__(
            self,
            model_name="tiny",
            ckpt_path=None,
            mask_id=32000,
            max_length=2048,
            batch_size=32,
            mc_num=256,
            padding=False,
            nll_type='mc',
            greddy=False,

            # gen
            cfg=0.,
            temp=0.1,
            steps=1024,
            gen_length=1024,
            block_length=1024,
            remasking='random',
            device="cuda",
            p=0.9
    ):
        super().__init__()
        assert nll_type in ['mc', 'chain_rule']
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self.device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device = torch.device(device)
            self._rank = 0
            self._world_size = 1

        model_name = f'Diff_LLaMA_{model_name}M'
        config = Config.from_name(model_name)
        self.model = TransEncoder(config).to(device)
        try:
            if ckpt_path.endswith('pth'):
                self.model.load_state_dict(remap_weights(torch.load(ckpt_path, weights_only=True)))
            elif ckpt_path.endswith('safetensors'):
                self.model.load_state_dict(remap_weights(load_file(ckpt_path)))
        except:
            self.model.load_state_dict(remap_weights(torch.load(ckpt_path, weights_only=True)['model']))

        self.model.eval()

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained('tinyllama_tokenizer')
        
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.padding = padding
        self.nll_type = nll_type
        self.greddy = greddy
        

        self.cfg = cfg
        self.temp = temp
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking    
        # self.device = torch.device(device)

        self.p = p

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_id, batch)

        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.padding:
            input = torch.full((batch.size(0), 2048), self.mask_id, device=self.device)
            input[:, :batch.shape[1]] = batch
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        '''
        Utilize the chain rule to compute the likelihood
        We need to perform len(target) forward passes in parallel
        '''
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2

        prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        mask_index = torch.triu(mask_index)

        perturbed_[mask_index] = self.mask_id
        perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        temp_index = torch.triu(temp_index, diagonal=1)
        mask_index[temp_index] = False
        logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().float()
        return loss


    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        '''
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.greddy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct


    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 2048

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                elif self.nll_type == 'chain_rule':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    
    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        # ds = ds.with_format("torch")


        
        batch_size = self.batch_size 
        length = len(ds)
        iters = length // batch_size if length % batch_size == 0 else length // batch_size + 1

        out = []
        for i in tqdm(range(iters), desc="Generating..."):
            end_index = (i + 1) * batch_size if (i + 1) * batch_size < length else length
            data = ds[i * batch_size: end_index]
            prompt = data["question_text"]
            prompt_lens = [len(x) for x in data['question']]

            stop_tokens = data["until"][0]

            generated_answer = diff_sample(self.model,
                             self.tokenizer,
                             prompt,
                             steps=self.steps,
                             context_length=256,
                             device=self.device,
                             topp=self.p)

            generated_answer = [ans[prompt_lens[i]:] for i, ans in enumerate(generated_answer)]
            
            generated_answer = self.tokenizer.batch_decode(generated_answer, skip_special_tokens=False)

            for i in range(len(generated_answer)):
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer[i]:
                        generated_answer[i] = generated_answer[i].split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.batch_decode(generated_answer_ids, skip_special_tokens=True)

            out += generated_answer

            self.accelerator.wait_for_everyone()

        return out


def sample_categorical(categorical_probs):
    gumbel_norm = -torch.rand_like(categorical_probs).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def top_p_sampling(probs, p=0.9):

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    probs.masked_fill_(indices_to_remove, 0)
    probs /= probs.sum(dim=-1, keepdim=True)
    index = sample_categorical(probs.to(torch.float64))
    
    return index

@ torch.no_grad()
def diff_sample(model, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., cfg_scale=2.,
                context_length=2048, eps=1e-5, dim=32000, device='cuda', topp=0.9):

    prompt = tokenizer(prompt, padding="longest", truncation=True, return_tensors="pt")['input_ids'].to(device)
    
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    for i in range(steps):
        mask_index = (x == dim)
        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = 1 - s / t if i < steps - 1 else 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            
        probs = logits.softmax(-1)
        x0 = top_p_sampling(probs.to(torch.float64), p=topp).int()

        x0 = torch.where(mask_index, x0, x)
        transfer_index = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device) < p_transfer
        x[transfer_index] = x0[transfer_index]

    return x

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()