import os
import argparse
import pathlib
import pickle
import copy
import numpy as np
import datetime
import pandas as pd
import contextlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import deepspeed
from deepspeed.utils import set_z3_leaf_modules
from transformers import LlamaTokenizer, MixtralForCausalLM, MixtralConfig, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP, MixtralSparseMoeBlock


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d_%H:%M:%S")


class SFTDataset(Dataset):
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return None


if __name__ == "__main__":
    MOE_LOSS_WEIGHT = 0.001

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--epoch-number", type=str)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.)
    parser.add_argument("--deepspeed-config-path", type=str)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()
    
    model_config = MixtralConfig.from_pretrained(args.model_path, router_aux_loss_coef=1e-3)
    model = MixtralForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to(torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    set_z3_leaf_modules(model, [MixtralSparseMoeBlock]) 
    model.train()

    deepspeed_config = json.load(open(args.deepspeed_config_path, "r"))
    engine, _, _, _ = deepspeed.initialize(
        config = MOCK_DS_CONFIG,
        model = model,
        dist_init_required = True
    )
    engine.train()

    train_dataset = SFTDataset(args.data_path)
    train_data_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42)
    train_dataloader = DataLoader(train_dataset, batch_size=accumulation_step, sampler=train_data_sampler)

    local_step = 0
    global_step = 0
    loss_step = 0
    accumulation_step = deepspeed_config.get("train_micro_batch_size_per_gpu", 1)
    for ep in range(args.epoch_number):
        for bid, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            loss = engine(**batch, use_cache=False).loss
            engine.backward(loss)
            engine.step()

            lm_loss_step += loss.item()
            local_step += 1

            if local_step % accumulation_step == 0:
                global_step += 1
                lm_loss_step /= accumulation_step
                if args.local_rank == 0:
                    print(f"time: {get_time_str()} step: {global_step} / {len(train_dataloader) // accumulation_step * args.epoch_number}, total_loss: {loss_step}")
                lm_loss_step = 0
                    
        model_config.save_pretrained(os.path.join(args.save_path, str(global_step)))
        tokenizer.save_pretrained(os.path.join(args.save_path, str(global_step)))
        engine.save_16bit_model(os.path.join(args.save_path, str(global_step)))