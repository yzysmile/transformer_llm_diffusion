# SFT(Supervised Fine Tune)
# Fine Tune和SFT本质上是一个东西， 都指用一些数据进行 fine tuning
# 区别：
#     1. LLM前时代一般指的是用domain数据进行微调，一般有具体任务；
#     2. LLM时代一般指的是instruction tuning。
# 参数高效（PEFT）的SFT实现的有效方式
# 包括. LoRA...

# 工作中一般将自己的数据集处理成相应的格式 与 开源数据集 混合
# 在此，我们只用开源的数据集
from datasets import load_dataset

test_dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train[:500]")

print(f"test_dataset:{test_dataset}")
# test_dataset:Dataset({
#     features: ['kind', 'input', 'target'],
#     num_rows: 500
# })
# 表示 数据集 test_dataset 由 500 行样本组成，并且有以下3个字段 'kind'、‘input'、'target'
# kind：数据类型（可能用于区分不同类型的任务）
# input：输入文本
# target：输出文本（期望的回答）

print(f"test_dataset.column_names:{test_dataset.column_names}")
# test_dataset.column_names:['kind', 'input', 'target']

print(f"test_dataset[100]:{test_dataset[100]}")
# test_dataset[100]:{'kind': 'ClassicalChinese', 'input': '我当时在三司，访求太祖、仁宗的手书敕令没有见到，然而人人能传诵那些话，禁止私盐的建议也最终被搁置。\n翻译成文言文：', 'target': '余时在三司，求访两朝墨敕不获，然人人能诵其言，议亦竟寝。'}

# 加载 Qwen/Qwen2.5-0.5B-Instruct 的 Tokenizer，用于文本预处理（分词、编码等）。
from transformers import  AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct'
)
# 默认情况下padding和truncation都是False

# inputs = tokenizer(
#     text,
#     padding=True,          # 自动填充到批次中最长序列，默认填充到右侧（padding_side='right'）
#     truncation=True,       # 自动截断到模型最大长度
#     max_length=512,        # 自定义截断长度（可选）
#     return_tensors="pt"    # 返回PyTorch张量
# )


# 要把数据处理成 与 fundation-model(qwen2-0.5b-instruct)的输入格式一致
def format_prompt(example):
    chat = [
        {"role":"system", "content":"你是一个由‘yzy’开发的非常棒的人工智能助手"},
        {"role":"user", "content":example["input"]},
        {"role":"assistant", "content":example["target"]}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

dataset = test_dataset.map(format_prompt, remove_columns=test_dataset.column_names)
# test_dataset.column_names = ["kind", "input", "target"]
# map() 会 遍历整个数据集，将每条数据传入 format_prompt 进行转换，并返回新的 text 字段。
# remove_columns=test_dataset.column_names 删除原始字段（"kind", "input", "target"），只保留 text

print(f"dataset:{dataset}")
# dataset:Dataset({
#     features: ['text'],
#     num_rows: 500
# })

print(f"dataset[100]:{dataset[100]}")
# dataset[100]:{'text': '<|im_start|>system\n你是一个由‘yzy’开发的非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n我当时在三司，访求太祖、仁宗的手书敕令没有见到，然而人人能传诵那些话，禁止私盐的建议也最终被搁置。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n余时在三司，求访两朝墨敕不获，然人人能诵其言，议亦竟寝。<|im_end|>\n'}


# 加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
                                             device_map='auto')

tokenizer.padding_side = 'left'

# LoRA 低秩适配
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32, # LoRA 低秩矩阵的秩，较大的 r 提高参数表达能力，但也增加计算量
    bias='none',
    task_type='CAUSAL_LM', # 任务类型，Causal LM (自回归语言模型)
    target_modules=['k_proj','v_proj','q_proj'] # 仅对 k_proj、v_proj、q_proj 进行微调
    # 还可以对 Q K V及FFN进行微调 lora微调参数量占原模型的0.1%~1%
    # 仅仅对QKV微调7B模型 0.36%
)

model = get_peft_model(model, peft_config)
model.config.use_cache = False  # 在模型配置里关掉 use_cache

# SFT 训练
from trl import SFTTrainer, SFTConfig

output_dir = './results'

training_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # batch_size 实际上为 1*4
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type='cosine', # 采用 余弦退火 (Cosine Annealing) 学习率调度
    num_train_epochs=1, # 只训练 1 个 epoch
    logging_steps=10, # 每 10 步打印一次日志
    fp16=True, # 16-bit 浮点训练
    gradient_checkpointing=True, # 启用梯度检查点，减少显存占用

    save_steps=15,  # 15 个 step 就保存一个 checkpoint，正常不会这么快
    max_steps=20,
)

trainer = SFTTrainer(
    model=model,
    args=training_config,  # ✅ 使用 SFTConfig
    dataset_text_field="text",
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# 保存微调模型
trainer.model.save_pretrained("./results/final-result")

# **打印可训练参数**
trainer.model.print_trainable_parameters()

# 合并 LoRA 参数
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "./results/final-result",
    device_map = 'auto',
)

merged_model = model.merge_and_unload()
# 通过合并和卸载 LoRA 模块，减少了额外模块的存储需求，避免了内存冗余。
# 一旦 LoRA 模块合并到主模型中，你就可以直接使用主模型进行推理，不需要再加载额外的 LoRA 模块。

# 模型推理
from transformers import pipeline
pipeline = pipeline(task='text-generation',
                    model=merged_model,
                    tokenizer=tokenizer)

prompt_example = """<|im_start|>system
你是一个由‘yzy’开发的非常棒的人工智能助手。<|im_end|>
<|im_start|>user
天气太热了，所以我今天没有学习一点。
翻译成文言文:<|im_end|>
<|im_start|>assistant
"""

print(pipeline(prompt_example, max_new_token=50)[0]["generated_text"])

# 工作中使用 LlamaFactory 的SFT流程，训练、推理和导出
# LlamaFactory 是 一个开源的 LLM（大语言模型）训练和推理框架，
# 支持 LLaMA、Mistral、Gemma、Baichuan 等多个模型的 微调（SFT）、对齐（DPO/RLHF）和推理，并且优化了 LoRA（低秩适配）、FlashAttention、P-tuning v2 等技术，以提高训练和推理效率。
# 它是 OptimalScale 团队开发的，目标是让 LLaMA 及其他大模型的微调和部署更加高效和易用，尤其适用于 开源 LLaMA 3、LLaMA 2、Mistral 等模型。

# llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
# llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
# llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml















