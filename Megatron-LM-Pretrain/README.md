# Mixtral训练框架

Forked from (Megatron-LM)[https://github.com/NVIDIA/Megatron-LM]

## 数据处理

数据路径格式：

```bash
DATA_ROOT_DIR
 |
 | - common crawl
 |    | - 0.jsonl
 |    | - 1.jsonl
 |    | - 2.jsonl
 |    | .......
 | - github
 |    | - 0.jsonl
 |    | - 1.jsonl
 |    | .......
```

以 common crawl 为例，可以使用如下命令得到分词后的语料。

```bash
python tools/preprocess_data.py \
       --input $DATA_ROOT_DIR/common_crawl \
       --output-prefix $ENCODED_DATA_ROOT_DIR/common_crawl \
       --json_keys text \
       --dataset-impl mmap \
       --tokenizer-type PretrainedTokenizerFromHF \
       --tokenizer-model  \
       --append-eod \
       --workers 8
```

分词结果是 `$ENCODED_DATA_ROOT_DIR/common_crawl_text_document.bin` 和 `$ENCODED_DATA_ROOT_DIR/common_crawl_text_document.idx`

## 模型处理

模型处理包括两部分：

+ 从 Huggingface 格式的模型参数转成 Megatron 格式的 checkpoint；
+ 从 Megatron 格式的 checkpoint 转成 Huggingface 格式的模型参数

### 从 Huggingface 转到 Megatron

把 Huggingface 格式的模型参数转成 Megatron 格式的 checkpoint 可以使用以下命令：

```bash
PYTHONPATH=$(pwd) python tools/checkpoint/util.py --model-type GPT \
--loader mixtral_hf \
--saver mixtral \
--load-dir $MIXTRAL_HUGGINGFACE_PATH \
--save-dir $MIXTRAL_MEGATRON_PATH \
--tokenizer-model $MIXTRAL_HUGGINGFACE_PATH/tokenizer.model \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP
```

8x7b 的 mixtral 在 4k 窗口上训练，推荐使用 16 卡，TP=4，PP=4

我们测试的是 4x7b 的 mixtral 模型在 4k 窗口上训练，使用 8 卡，TP=4，PP=2

## 从 Megatron 转到 Huggingface

```bash
PYTHONPATH=. python tools/checkpoint/convert_megatron_to_hf_mixtral.py \
	--megatron-checkpoint-path $MIXTRAL_MEGATRON_PATH \
	--huggingface-config-path $MIXTRAL_HUGGINGFACE_PATH \
	--huggingface-checkpoint-path $MIXTRAL_HUGGINGFACE_PATH \
	--tensor-parallel-size $TP --pipeline-parallel-size $PP
```

## 训练脚本

我们提供了 `mixtral_4x7b_example.sh` 作为训练脚本示例。

修改脚本的 `CHECKPOINT_PATH` 为 `megatron-checkpoint-path`，`BLENDED_DATA` 为可带权重（可选）的预处理数据路径（如 `0.6 ENCODED_DATA_ROOT_DIR/common_crawl_text_document`）运行脚本开始训练。

```bash
bash mixtral_4x7b_example.sh
```



