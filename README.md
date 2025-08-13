# InternLM SFT 训练项目

基于 InternLM-7B 模型的监督微调（SFT）项目，使用 LoRA 技术进行高效训练。

## 项目结构

```
internlm-sft/
├── train_sft.py          # 主训练脚本
├── test.py               # 测试脚本
├── app.py                # 应用脚本
├── internlm-7b-model/    # 预训练模型目录
├── hz_sft_datav2/        # 训练数据目录
├── output_refusev2/      # 训练输出目录
└── readme.md             # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets

## 安装依赖

```bash
pip install torch transformers peft datasets accelerate
```

## 使用方法

### 1. 训练模型

```bash
python train_sft.py \
    --model_name_or_path ./internlm-7b-model \
    --use_lora true \
    --data_path ./hz_sft_datav2 \
    --output_dir ./output_refusev2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 4e-4
```

### 2. 使用 VS Code 调试

在 `.vscode/launch.json` 中配置了调试配置，可以直接在 VS Code 中运行训练。

## 配置说明

- **LoRA 配置**：使用 LoRA 进行参数高效微调
- **数据集**：支持 instruction-following 格式的数据
- **训练策略**：支持 DeepSpeed 和普通训练模式

## 注意事项

- 模型文件较大，请确保有足够的存储空间
- 训练数据需要按照特定格式准备
- 建议使用 GPU 进行训练

## 许可证

MIT License
