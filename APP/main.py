from fastapi import FastAPI
# CausalLM,因果语言模型 
# Auto 自动适配，AutoTokenizer自动检测模型类型，加载对应的分词器,AutoModelForCausalLM自动适配不同架构的因果语言模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# 模型路径
model_path = "/root/autodl-tmp/Models/deepseek-r1-1.5b-merged"

# 加载 tokenizer （分词器）
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型并移动到可用设备（GPU/CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# ("/generate")表示这个接口的访问路径，也就是说可通过http://服务器地址：端口/generate来访问这个接口
# 当客户端向服务器发送一个get请求到/generate路径，并且传入prompt参数时，fastapi会自动被这个装饰器修饰的generate_text函数
# http://localhost:8000/generate?prompt=你好，请介绍一下自己 
# 接口暴露规则：一个装饰器=一个接口
# 在浏览器中直接输入http://localhost:8000/generate?prompt=你好
@app.get("/generate")
async def generate_text(prompt: str):
    # 使用 tokenizer 编码输入的 prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 使用模型生成文本
    outputs = model.generate(inputs["input_ids"], max_length=150)
    
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": generated_text}

