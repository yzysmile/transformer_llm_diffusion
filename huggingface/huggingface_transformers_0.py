# transformers 是最常用的库之一，支持多种预训练模型（如GPT、BERT、T5等）并提供了方便的API进行微调。
# deepspeed 主要用于大规模分布式训练，它通过优化内存使用和加速训练过程，特别适用于需要超大显存的模型。
# peft 主要用于参数高效微调（例如LoRA），可以在不大幅增加参数量的情况下有效地微调大模型。
# flash attention 通过加速Transformer中的注意力机制，显著提高了训练速度，特别是在使用较大模型和大规模数据时。

# lamini，它是一个用于大语言模型精调的框架，简化了调整和微调大模型的过程，但在广泛使用的基础库中，transformers 和 deepspeed 更为常见。
import os

os.environ["HF_HOME"] = "./huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

# 清空 GPU 缓存
torch.cuda.empty_cache()
gc.collect()

# step1.加载模型 和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map = 'cuda',
    torch_dtype= "auto",
    trust_remote_code=True
)
# print(model)

tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct'
)

print(tokenizer.special_tokens_map)
# {'eos_token': '<|im_end|>',
# 'pad_token': '<|endoftext|>',
# 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}
# 一句话的最后一个token是"im_end"，如果需要补位，其补位（pad）的都是“endoftext”。


input_ids = tokenizer('<|image_pad|>')
print(input_ids)
# {'input_ids': [151655], 'attention_mask': [1]}   # 'attention_mask': [1]表示这是一个有效的token
# 模型通常需要 固定长度的输入，但句子长度可能不同。
# 因此，需要用 填充 token（padding token）补齐较短的句子，以匹配最长句子的长度。


#{'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>',
# special_tokens包括
# ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}

# step2.构建prompt 并 编码
prompt = "讲一个猫有关的笑话？"

# 正确写法：用方括号 [] 定义列表
messages = [
    {"role": "system", "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 在做instruct tuning的时候，要把sft的数据构造成 模板的形式
text = tokenizer.apply_chat_template(
    messages,
    tokenizer=False, # 为False时， message不会进行tokenization
    add_generation_prompt=True  # 最后多出 "<|assistant|>\n"，模型知道要在这个地方开始生成。
)
# 如果 text 是 token ID（list），则尝试解码回字符串
if isinstance(text, list):
    text = tokenizer.decode(text, skip_special_tokens=True)

print(f"text:{text}")
# text:system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
# user
# 讲一个猫有关的笑话？
# assistant

# step3. 编码 llm推理时 输入的是文本token，训练时需要embedding
# 推理、解码
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
# model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
print(f"model_inputs: {model_inputs}")
# model_inputs: {'input_ids': tensor([[  8948,    198,   2610,    525,   1207,  16948,     11,   3465,    553,
#           54364,  14817,     13,   1446,    525,    264,  10950,  17847,    624,
#             872,    198,  99526,  46944, 100472, 101063,   9370, 109959,  94432,
#           77091,    198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1]], device='cuda:0')}
# model_inputs是一个 字典 dict，两个key分别是 "input_ids","attention_mask"
# 对应的value分别是 text token化后的结果 以及 是否有效的token
# ids 是指 token IDs，也就是 词汇表（vocabulary）中每个单词或子词的索引

generated_output = model.generate(
    **model_inputs,
    # "**"将 dict 拆分为key-value的形式，
    # 等价于 input_ids=model_inputs["input_ids"],  attention_mask=model_inputs["attention_mask"]
    max_new_tokens=512
)
print(f"generated_output:{generated_output}")


# 从 generated_output 中提取新生成的 token ID，并去掉输入的 token ID 部分。它用于 截取模型生成的新内容，而不包含输入 prompt。
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs["input_ids"], generated_output)
]
print(f"generated_ids:{generated_ids}")

responses = tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]
print(f"response: {responses}")

#"======================================================================"

# 使用transformers的pipeline简化流程
from transformers import pipeline

# step1: 生成pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)
# do_sample=True：使用随机采样，生成的文本是非确定性的

# step2： 构建prompt
messages = [
    {"role":"user", "content":"写一个和猫有关的笑话。"}
]

# step3: 输出并解码
output = generator(messages)
print(f"output:{output}")

print(output[0]["generated_text"])
# output返回的是一个列表，
# output【0】是一个dict，【“generated_text”】是dict的key