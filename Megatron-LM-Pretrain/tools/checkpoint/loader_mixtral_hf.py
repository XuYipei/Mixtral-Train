# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types
import torch
import transformers
from tqdm import tqdm


def add_arguments(parser):
    group = parser.add_argument_group(title='Mixtral 8x7B HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    if major < 4 or minor < 31:
        raise ValueError("the version transformers should greater or equal 4.31")


def load_args_from_checkpoint(args):

    # Read Mixtral args.
    args_path = os.path.join(args.load, "config.json")
    with open(args_path) as f:
        mixtral_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = mixtral_args.get("max_position_embeddings", 32768)
    args.num_experts = mixtral_args.get("num_local_experts", None)
    args.num_experts_per_tok = mixtral_args.get("num_experts_per_tok", 2)
    args.moe_type = "megatron"
    args.moe_grouped_gemm = True
    args.moe_router_topk = 2
    args.hidden_size = mixtral_args["hidden_size"]
    args.num_attention_heads = mixtral_args["num_attention_heads"]
    args.num_layers = mixtral_args["num_hidden_layers"]
    args.global_batch_size = 1
    args.norm_epsilon = mixtral_args["rms_norm_eps"]
    args.iteration = 1
    args.position_embedding_type = "rope"
    args.swiglu = True
    args.tokenizer_type = "Llama2Tokenizer"
    args.bf16 = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.add_position_embedding = False
    args.untie_embeddings_and_output_weights = True
    args.vocab_size = mixtral_args["vocab_size"]
    args.padded_vocab_size = mixtral_args["vocab_size"]
    args.mixtral = mixtral_args
    args.ffn_hidden_size = mixtral_args["intermediate_size"]
    args.gradient_accumulation_fusion = False

    if "num_key_value_heads" in mixtral_args:
        args.group_query_attention = True
        args.num_query_groups = mixtral_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params."""
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params."""

    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.query_key_value.weight.data.copy_(torch.cat([ 
        hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.dense.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer, layer_idx):
    """Set Mixtral SMOE params."""

    mlp = layer.mlp
    hf_mlp = hf_layer.block_sparse_moe

    gate = mlp.router.weight.data.copy_
    hf_gate_weight = hf_mlp.gate.weight
    gate(hf_gate_weight)

    w1s = []
    w2s = []
    for idx in range(args.num_experts):
        hf_w1_weight = getattr(hf_mlp.experts, f"{idx}").w1.weight
        hf_w2_weight = getattr(hf_mlp.experts, f"{idx}").w2.weight
        hf_w3_weight = getattr(hf_mlp.experts, f"{idx}").w3.weight
        w1s.append(torch.cat([hf_w1_weight.transpose(0, 1), hf_w3_weight.transpose(0, 1)], dim=-1))
        w2s.append(hf_w2_weight.transpose(0, 1))
    hf_w1 = torch.cat(w1s, dim=-1)
    hf_w2 = torch.cat(w2s, dim=0)

    mlp.experts.weight1.data.copy_(hf_w1)
    mlp.experts.weight2.data.copy_(hf_w2)




def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer, layer_idx)
    layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(args):
    """Set model params."""
    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM

    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(args.load, device_map="cpu")

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)


    print(model)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):

    # Mixtral requires HF transformers >=4.36.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model

    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    if not args.model_type == 'GPT':
        raise ValueError("Mixtral is a GPT model.")
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.moe_type = margs.moe_type
    md.moe_grouped_gemm = margs.moe_grouped_gemm
    md.moe_router_topk = margs.moe_router_topk
    md.num_experts = margs.num_experts
    md.num_experts_per_tok = margs.num_experts_per_tok
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.add_position_embedding = margs.add_position_embedding
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    
    md.sequence_parallel = 1
    md.expert_model_parallel_size = 1

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    else:
        if hasattr(model.language_model.embedding, 'position_embeddings'):
            raise ValueError("model should have position_embeddings")

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.encoder.layers[layer_num]
        message["input norm weight"] = layer.input_norm.weight.data
        message["post norm weight"] = layer.post_attention_norm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_router_weight = []
        mlp_gate_bias = []

        layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.query_key_value.weight.data)
        dense_weight.append(layer.self_attention.dense.weight.data)
        mlp_router_weight.append(layer.mlp.router.weight.data)

        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            mlp_gate_bias.append(layer.mlp.gate.bias.data)

        # for expert_idx in range(margs.num_experts):
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            mlp_experts_weight1 = torch.chunk(layer.mlp.experts.weight1.data, margs.num_experts, dim=-1)
            message["mlp experts weight1 W"] = torch.cat([x.chunk(2, dim=-1)[0] for x in mlp_experts_weight1], dim=-1)
            message["mlp experts weight1 V"] = torch.cat([x.chunk(2, dim=-1)[1] for x in mlp_experts_weight1], dim=-1)
        else:
            message[f"mlp experts weight1"] = layer.mlp.experts.weight1.data
        message[f"mlp experts weight2"] = layer.mlp.experts.weight2.data

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp router weight"] = torch.cat(mlp_router_weight, dim=1)

        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            message["mlp gate bias"] = torch.cat(mlp_gate_bias, dim=0)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.language_model.encoder.final_norm.weight.data,
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.language_model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise