# Mixtral 微调代码

## 环境配置

```
deepspeed == 0.13.5 (>= 0.13.2)
torch == 2.1.2 (> 2.0.x)
```

## 代码运行

我们提供了使用 deepspeed 的 Mixtral 模型微调脚本 `train.py`，需要在代码中修改成自己的 SFTDataset 方可使用。

对于 4x7b MoE 模型，我们推荐使用 8 卡进行微调，可以使用 `deepspeed_config.json` 作为 deepspeed 的示例运行设置。

代码运行命令：

```
deepspeed train.py \
    --model-path $MIXTRAL_HUGGINGFACE_PATH \
    --save-path $MIXTRAL_FINETUNE_SAVE_PATH \
    --data-path $FINETUNE_DATA_PATH \
    --epoch-number $FINETUNE_EPOCH_NUMBER \
    --router-aux-loss-coef $MOE_ROUTER_AUX_LOSS_COEF \
    --deepspeed-config-path ./deepspeed_config.json
```