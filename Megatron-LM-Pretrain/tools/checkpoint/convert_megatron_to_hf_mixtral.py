import torch
import argparse
import os
from tqdm import tqdm

from transformers import MixtralForCausalLM, MixtralConfig


parser = argparse.ArgumentParser()
parser.add_argument('--megatron-checkpoint-path', type=str, default=None)
parser.add_argument('--huggingface-config-path', type=str, default=None)
parser.add_argument('--huggingface-checkpoint-path', type=str, default=None)
parser.add_argument('--tensor-parallel-size', type=int, default=1)
parser.add_argument('--pipeline-parallel-size', type=int, default=1)
args = parser.parse_args()


latest_iteration_step = open(os.path.join(args.megatron_checkpoint_path, "latest_checkpointed_iteration.txt")).read()

mix_model_config = MixtralConfig.from_pretrained(args.huggingface_config_path)
mix_model = MixtralForCausalLM(mix_model_config).to(torch.bfloat16)


for pipeline_parallel_rank in range(args.pipeline_parallel_size):
    ckpts = []
    for tensor_parallel_rank in range(args.tensor_parallel_size):
        if args.pipeline_parallel_size > 1:
            ckpts.append(torch.load(os.path.join(args.megatron_checkpoint_path, f"iter_{latest_iteration_step.zfill(7)}", f"mp_rank_{str(tensor_parallel_rank).zfill(2)}_{str(pipeline_parallel_rank).zfill(3)}", "model_optim_rng.pt"), map_location="cpu"))
        else:
            ckpts.append(torch.load(os.path.join(args.megatron_checkpoint_path, f"iter_{latest_iteration_step.zfill(7)}", f"mp_rank_{str(tensor_parallel_rank).zfill(2)}", "model_optim_rng.pt"), map_location="cpu"))
        # ckpts.append(torch.load(f"./ckpts/mixtral_4x7b/iter_0066560/mp_rank_0{tp}_000/model_optim_rng.pt", map_location="cpu"))

    if pipeline_parallel_rank == 0:
        mix_model.model.embed_tokens.weight.data.copy_(torch.cat([ckpt["model"]["language_model"]["embedding"]["word_embeddings"]["weight"] for ckpt in ckpts], dim=0))
    elif pipeline_parallel_rank == args.pipeline_parallel_size - 1:
        mix_model.lm_head.weight.data.copy_(torch.cat([ckpt["model"]["language_model"]["output_layer"]["weight"] for ckpt in ckpts], dim=0))

    begin_later_idx = (mix_model_config.num_hidden_layers // args.pipeline_parallel_size) * pipeline_parallel_rank
    for layer_idx in tqdm(range(mix_model_config.num_hidden_layers // args.pipeline_parallel_size)):     
        q_projs = []
        k_projs = []
        v_projs = []
        o_projs = []
        for ckpt in ckpts:
            qkv_weight = ckpt["model"]["language_model"]["encoder"][f"layers.{layer_idx}.self_attention.query_key_value.weight"]
        
            nh = mix_model_config.num_attention_heads // args.tensor_parallel_size
            ng = mix_model_config.num_key_value_heads // args.tensor_parallel_size
            dim = mix_model_config.hidden_size // mix_model_config.num_attention_heads

            qkv_weight = qkv_weight.reshape(ng, -1, mix_model_config.hidden_size)
            q_proj, k_proj, v_proj = qkv_weight.split([dim * nh // ng, dim, dim], dim=1)

            q_projs.append(q_proj.reshape(-1, mix_model_config.hidden_size))
            k_projs.append(k_proj.reshape(-1, mix_model_config.hidden_size))
            v_projs.append(v_proj.reshape(-1, mix_model_config.hidden_size))
            
            o_projs.append(ckpt["model"]["language_model"]["encoder"][f"layers.{layer_idx}.self_attention.dense.weight"])
        
        mix_model.model.layers[begin_later_idx + layer_idx].self_attn.q_proj.weight.data.copy_(torch.cat(q_projs, dim=0))
        mix_model.model.layers[begin_later_idx + layer_idx].self_attn.k_proj.weight.data.copy_(torch.cat(k_projs, dim=0))
        mix_model.model.layers[begin_later_idx + layer_idx].self_attn.v_proj.weight.data.copy_(torch.cat(v_projs, dim=0))
        mix_model.model.layers[begin_later_idx + layer_idx].self_attn.o_proj.weight.data.copy_(torch.cat(o_projs, dim=1))

        mix_model.model.layers[begin_later_idx + layer_idx].input_layernorm.weight.data.copy_(ckpts[0]["model"]["language_model"]["encoder"][f"layers.{layer_idx}.input_norm.weight"])

        mix_model.model.layers[begin_later_idx + layer_idx].block_sparse_moe.gate.weight.data.copy_(ckpts[0]["model"]["language_model"]["encoder"][f"layers.{layer_idx}.mlp.router.weight"])
        w1s = [[] for _ in range(mix_model_config.num_local_experts)]
        w2s = [[] for _ in range(mix_model_config.num_local_experts)]
        w3s = [[] for _ in range(mix_model_config.num_local_experts)]
        for ckpt in ckpts:
            weight1 = ckpt["model"]["language_model"]["encoder"][f"layers.{layer_idx}.mlp.experts.weight1"]
            weight2 = ckpt["model"]["language_model"]["encoder"][f"layers.{layer_idx}.mlp.experts.weight2"]
            weight1 = weight1.view(mix_model_config.num_local_experts, mix_model_config.hidden_size, -1)
            weight2 = weight2.chunk(mix_model_config.num_local_experts, dim=0)
            for e in range(mix_model_config.num_local_experts):
                w1, w3 = weight1[e].chunk(2, dim=-1)
                w1s[e].append(w1)
                w3s[e].append(w3)
                w2s[e].append(weight2[e])
        for e in range(mix_model_config.num_local_experts):
            mix_model.model.layers[begin_later_idx + layer_idx].block_sparse_moe.experts[e].w1.weight.data.copy_(torch.cat(w1s[e], dim=-1).transpose(0, 1))
            mix_model.model.layers[begin_later_idx + layer_idx].block_sparse_moe.experts[e].w2.weight.data.copy_(torch.cat(w2s[e], dim=0).transpose(0, 1))
            mix_model.model.layers[begin_later_idx + layer_idx].block_sparse_moe.experts[e].w3.weight.data.copy_(torch.cat(w3s[e], dim=-1).transpose(0, 1))

        mix_model.model.layers[begin_later_idx + layer_idx].post_attention_layernorm.weight.data.copy_(ckpts[0]["model"]["language_model"]["encoder"][f"layers.{layer_idx}.post_attention_norm.weight"])


mix_model.save_pretrained(args.huggingface_checkpoint_path, max_shard_size="5GB")