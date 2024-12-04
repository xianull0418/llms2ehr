from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
import torch
from test_dataset_twice import LocalJsonDataset  # 引入自定义数据集加载模块

# 设置模型的最大序列长度
max_seq_length = 1024

# 加载预训练模型和分词器
model_name = "Qwen/Qwen2-0.5B"  
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
# print(f"千问模型参数：{model.config}")


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 配置 LoRA 参数并应用
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# print(f"LoRAmodel参数：{model.config}")
for name, param in model.named_parameters():
    print(name)
# 加载和预处理多个 JSON 文件数据集
data_dir = "/home/jcw25/Project/llmsGemr/dataset"  # 数据集文件夹路径
custom_dataset = LocalJsonDataset(
    json_dir=data_dir,  # 确保此处为包含多个 JSON 文件的文件夹路径
    tokenizer=tokenizer,
    max_seq_length=max_seq_length
)
dataset = custom_dataset.get_dataset()

# 将数据分成训练集和验证集
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

def preprocess_function(examples):
    # 分别对每个输入部分进行分词，不进行填充和截断
    inputs_1 = tokenizer(
        examples["inputs_1_str"],
        truncation=False,
        padding=False,
    )
    inputs_2 = tokenizer(
        examples["inputs_2_str"],
        truncation=False,
        padding=False,
    )
    inputs_3 = tokenizer(
        examples["inputs_3_str"],
        truncation=False,
        padding=False,
    )
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for ids1, ids2, ids3, label_text in zip(
            inputs_1["input_ids"],
            inputs_2["input_ids"],
            inputs_3["input_ids"],
            examples["outputs_str"]
        ):
        # 拼接输入序列
        ids = ids1 + ids2 + ids3
        # 截断到最大长度
        ids = ids[:max_seq_length]
        # 创建对应的 attention_mask
        masks = [1] * len(ids)
        
        # 对标签进行分词，截断到与输入序列相同的长度
        label_ids = tokenizer(
            label_text,
            max_length=len(ids),
            truncation=True,
            padding="max_length",
        )["input_ids"]
        
        # 如果标签长度不足，进行填充
        if len(label_ids) < len(ids):
            padding_length = len(ids) - len(label_ids)
            label_ids += [tokenizer.pad_token_id] * padding_length
        else:
            label_ids = label_ids[:len(ids)]
        
        # 对输入序列进行填充
        if len(ids) < max_seq_length:
            padding_length = max_seq_length - len(ids)
            ids += [tokenizer.pad_token_id] * padding_length
            masks += [0] * padding_length
            label_ids += [tokenizer.pad_token_id] * padding_length
        
        input_ids.append(ids)
        attention_mask.append(masks)
        labels.append(label_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }



# 对训练和验证集进行分词预处理
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["inputs_1", "inputs_2", "inputs_3", "outputs"])
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["inputs_1", "inputs_2", "inputs_3", "outputs"])

# 设置训练参数
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_steps=20,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    weight_decay=0.01,
    report_to="none",
    seed=3407,
    logging_dir="./logs",
    eval_accumulation_steps=8
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)


# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")



