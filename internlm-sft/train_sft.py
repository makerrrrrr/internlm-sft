from peft.tuners.lora import LoraLayer
import copy
import logging
import os
import torch
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List

# 创建日志记录器
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    source_length: int = field(default=512)
    target_length: int = field(default=512)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    # 加载数据集
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    # dataset
    # Dataset({
    #     features: ['instruction', 'input', 'output'],
    #     num_rows: 1436866
    # })
    return raw_datasets


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [   #tokenizer返回一个字典，包含input_ids,attention_mask,token_type_ids等字段,input_ids和attention_mask是tokenizer的标准输出，不同模型可能会有额外的字段
        tokenizer(
            text,
            return_tensors="pt",#这些字段都是pytorch张量
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")
    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    # shuffle(seed=42)现将数据集随机打乱（seed固定，结果可复现）
    # select(range(min(len(dataset), 100))) 选择前100条数据;select按行进行索引
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), 3)))
    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['instruction']
        if 'input' not in examples.keys():
            input_data = [""] * len(ins_data)
        else:
            input_data = examples['input']
        output = examples['output']

        len_ = len(ins_data)

        # sources = []
        # targets = []

        # for i in range(len_):
        #     s_t = prompt_input.format_map({'instruction':ins_data[i],
        #                                    'input':input_data[i]}) if input_data[i] != "" else prompt_input.format_map({'instruction':ins_data[i]})
        #     sources.append(s_t)

        sources = [prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]}) if input_data[
            i] != "" else prompt_no_input.format_map(
            {'instruction': ins_data[i]})
            for i in range(len_)]
        sources = [i[:data_args.source_length] for i in sources]
        targets = [
            f"{example[:data_args.target_length-1]}{tokenizer.eos_token}" for example in output]

        # sources = [prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #            for example in examples]
        # targets = [
        #     f"{example['output']}{tokenizer.eos_token}" for example in examples]

        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    # 数据映射函数
    dataset = dataset.map(
        
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=1
    ).shuffle()
    return dataset



def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:
    
    print(f"🔍 模型路径: {model_args.model_name_or_path}")
    print(f"🔍 路径是否存在: {os.path.exists(model_args.model_name_or_path)}")
    print(f"🔍 开始加载模型...")
    
    if training_args.use_deepspeed:

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True

        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # device_map='auto',用于推理时的多GPU分配
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True

        )

    if model_args.use_lora:

        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model
        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "o_proj","gate_proj", "down_proj", "up_proj"
        ]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    return model, tokenizer


def train():
    # 1.创建解析器，传入dataclass类
    # 将命令行参数自动映射到这些数据类的实例中
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    # 2.解析命令行参数，返回dataclass实例
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 3.加载模型和tokenizer
    # 模型和分词器一起加载(自定义)，确保模型和分词器必须匹配
    # 相比于from_pretrained()-底层加载方法:直接加载模型权重和配置,一次只能加载一个组件
    # load_model_and_tokenizer-高层封装函数:加载模型、分词器、应用LoRA等配置
    model, tokenizer = load_model_and_tokenizer(
        model_args, training_args, data_args)


    with training_args.main_process_first(desc="loading and tokenization"):

        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()
