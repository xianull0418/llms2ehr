# run_inference.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 设置模型的最大序列长度（保持与训练时一致）
max_seq_length = 1024  

# 基础模型名称或路径（确保与训练时一致）
base_model_name = "Qwen/Qwen2-0.5B"

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True  # 添加 trust_remote_code
)

# 加载分词器（与训练时相同）
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, 
    use_fast=False, 
    trust_remote_code=True  # 添加 trust_remote_code
)
tokenizer.pad_token = tokenizer.eos_token

# 加载微调后的 LoRA 模型路径
lora_model_path = "/home/jcw25/Project/just_train/just_train/lora_model"  

# 加载微调后的 LoRA 模型
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True  # 添加 trust_remote_code
)

# 将模型设置为评估模式
model.eval()

def generate_answer(input1, input2, input3):    
    """
    根据三个输入生成答案。

    参数:
    - input1: str，第一个输入文本
    - input2: str，第二个输入文本
    - input3: str，第三个输入文本

    返回:
    - str，生成的答案
    """
    # 构建输入文本，参考训练时的输入格式
    input_text = (
        f"入院记录: {input1}\n"
        f"首次病程记录: {input2}\n"
        f"上级医师查房记录: {input3}\n"
        f"出院小结:"
    )
    
    # 分词
    inputs = tokenizer(
        [input_text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_seq_length,
    )
    # 确保输入张量在模型的同一设备上
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,         # 根据需要调整生成的最大新 token 数
            do_sample=True,             # 启用采样以生成更具多样性的答案
            temperature=0.7,            # 控制采样的温度
            top_p=0.9,                  # 使用核采样，考虑累积概率超过 top_p 的 token
            top_k=50,                   # 只考虑概率最高的 top_k 个 token
            eos_token_id=tokenizer.eos_token_id,  # 设置结束标记
        )
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 如果需要，可以根据实际情况调整截断逻辑
    return decoded_output.strip()

if __name__ == "__main__":
    print("请输入您的三个输入内容，输入 'exit' 退出。")
    while True:
        print("\n--- 输入部分 1 ---")
        input1 = input("请输入第一个输入（例如：入院记录）: ")
        if input1.lower() == 'exit':
            print("程序已退出。")
            break

        print("\n--- 输入部分 2 ---")
        input2 = input("请输入第二个输入（例如：首次病程记录）: ")
        if input2.lower() == 'exit':
            print("程序已退出。")
            break

        print("\n--- 输入部分 3 ---")
        input3 = input("请输入第三个输入（例如：上级医师查房记录）: ")
        if input3.lower() == 'exit':
            print("程序已退出。")
            break

        # 生成答案
        answer = generate_answer(input1, input2, input3)
        print("\n--- 生成的答案 ---")
        print(answer)
