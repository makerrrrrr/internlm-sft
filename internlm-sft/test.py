import os
import logging
import transformers
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import DataCollatorForSeq2Seq,Trainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata=({"help": "Path to the training data."})
        source_length: int=field(default=512)
        target_length: int=field(default=512)
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata=(
            {"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    )
    use_deepspeed: bool = field(default=False)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"
            all_file_list.append(standard_path)

    return all_file_list


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:
    print(f"🔍 模型路径: {model_args.model_name_or_path}")
    print(f"🔍 路径是否存在: {os.path.exists(model_args.model_name_or_path)}")
    print(f"🔍 开始加载模型...")
    if training_args.use_deepspeed:
        model=transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True
        )
    else:
        model=transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True
        )
    
    if model_args.use_lora:
        logging.warning("Loading model to Lora")
        from peft import LoraConfig,get_peft_model
        LORA_R=32
        LORA_DROPOUT=0.05
        TARGET_MODULES=[
            "o_proj","gate_proj","down_proj","up_proj"
        ]
        config=LoraConfig(
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model=get_peft_model(model,config)
        model.print_trainable_parameters()
    tokenizer=transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    return model,tokenizer
    


def load_dataset_from_path(data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")
    dataset = load_dataset_from_path(
        data_path=data_path,
    )


def train():
    # 1.创建解析器，传入dataclass类
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    # 2.解析命令行参数，返回dataclass实例
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 3.加载模型和tokenizer
    # 模型和分词器一起加载，确保模型和分词器必须匹配
    # 相比于from_pretrained()-底层加载方法:直接加载模型权重和配置,一次只能加载一个组件
    model, tokenizer = load_model_and_tokenizer(
        model_args, training_ags, data_args
    )

    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = make_train_dataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,label_pad_token_id=IGNORE_INDEX)
