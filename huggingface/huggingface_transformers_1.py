# 1. 了解transformers llm 不同输入和输出的区别？ 以及看看不同的output 可以有什么用？
# 答： 大模型的输入需要遵循一定的格式，e.g. chatML格式；

# 2. 了解RMSNorm、layerNorm、batchNorm的区别?


# 3. 了解KV cache的原理， 以及在推理时怎么使用？
#    generation_output = model.generate(
#      input_ids=input_ids,
#      max_new_tokens=1000,
#      use_cache=True # 或者 use_cache=False
#    )
# 答：为了提高推理效率，尤其是在处理长文本时，
#     KV Cache（Key-Value Cache）被用来缓存模型中计算得到的 键（Key） 和 值（Value），从而避免重复计算，从而“加速推理”。
#     KV Cache只能用于Decoder架构的模型，这是因为Decoder中的attention存在Causal Mask,在推理时前面已经生成的字符不与后面的字符产生attention

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc #  Garbage Collection（垃圾回收）

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "cuda",
    torch_dtype= "auto",
    trust_remote_code=True
)
# “trust_remote_code=True” 表示信任模型的来源，允许执行该模型的 自定义代码（例如模型的预处理代码、特定的解码器等）
#  即如果模型需要特定的代码来初始化或处理，你允许从远程服务器加载并执行这些代码。

# 清空 GPU 缓存
torch.cuda.empty_cache()
gc.collect()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text = False,
    max_new_tokens = 50,
    do_sample=False,
)

prompt = "《静夜思》的作者是谁？" # 这不是 chatML 格式
output = generator(prompt)

print(output[0]["generated_text"])

# chatML 格式
messages = [
    {"role":"system","content":"你是一个很有用的助手"},
    {"role":"user", "content":prompt},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"text:{text}")
output = generator(text)[0]["generated_text"]
print(f"output:{output}")

prompt = "The capital of France is"
# model包含3个重要的方法，包含 “.generate”、“.model”、“.lm_head”
# .generate:让模型 迭代预测下一个 token 直到 max_length 或 EOS（结束标记）
# .model: 用于提取 隐藏状态 (hidden states)
# .lm_head: 语言模型的输出层，它把 hidden_states 转换为词汇表概率 （即 vocab_size 维度）。

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

model_output = model.model(model_inputs["input_ids"])
print(model_output[0].shape)  # torch.Size([1, 5, 896]); 896是 hidden_status

lm_head_output = model.lm_head(model_output[0])
print(lm_head_output.shape) # torch.Size([1, 5, 151936]); 151936是 vocab_size

prediction_token_id = lm_head_output[0, -1].argmax(-1)
# lm_head_output[0, -1] 取出batch内第一个样本的最后一个token位置的输出，形状为 (vocab_size,) = (151936,)
# argmax(-1)表示取出(151936,)中概率最大的索引

print(f"prediction_token_id:{prediction_token_id}")
decode_prediction_token_id = tokenizer.decode(prediction_token_id)
print(decode_prediction_token_id) # "Paris"

