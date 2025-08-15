# 1. 视觉理解能力
# 说明：不仅能识别常见物体（花、鸟、鱼、虫），还能理解图像中的 文字、图表、图标、图形、布局。
# 用途示例：识别图像中的说明文字、阅读图表、理解 PPT 页面排版等。

# 2. Agentic 能力
# 说明：可以作为一个“视觉智能体”使用，具备 推理 和 指挥工具的能力。
# 用途示例：模拟点击按钮、填写表单、操作网页等（Agent 场景）。

# 3. 长视频理解 + 事件捕捉
# 说明：可以理解长达 1 小时以上的视频，并且能定位其中的事件片段。
# 用途示例：用户问“视频里什么时候有人摔倒？”模型能指出相关时间段。

# 4. 视觉定位（多种格式）
# 说明：可以通过 **生成框（bounding box）或点（point）**来定位图像中的物体，支持稳定输出格式（如 JSON）。
# 用途示例：生成如 { "label": "cat", "bbox": [x1, y1, x2, y2] } 的结果。

# 5. 结构化输出能力
# 说明：能对发票、表单、表格等图像中的结构化信息进行解析并输出。
# 用途示例：识别发票上的“发票号、金额、开票单位”等字段，输出为 JSON 或 CSV 格式。


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import gc

# 清空 GPU 缓存
torch.cuda.empty_cache()
gc.collect()

# 加载模型
    # 根据设备自动选择 float32、float16、或 bfloat16
    # 自动将模型分布在 GPU / CPU 上
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 自动加载用于图像 + 文本输入的 tokenizer、image processor、video processor 等
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# 包含了 image_processor、tokenizer、video_processor
    # tokenizer: 把文本转成token序列的ID
    # image_processor: PIL.Image 自动 resize、normalize 等 并 tensor
    # video_processor: 视频帧列表 → tensor， 输出 [batch, frames, channels, H, W] 结构

# tokenizer
tokenizer = processor.tokenizer
pad_id = tokenizer.pad_token_id # '<|endoftext|>'对应的token id
pad = tokenizer.pad_token # '<|endoftext|>'
    # 打印所有 special tokens 及它们的 ID
# print(tokenizer.special_tokens_map)
    # {'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 
    # 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}
    # 这些token本身不会显示给用户，但会让模型在训练时学会“这里的内容是特定类型”，从而在推理时正确解析和生成对应的多模态信息。

tokens = tokenizer("描述图像内容", return_tensors="pt")  
    # 输出一个字典 其中包含2个键 "input_ids", "attention_mask"
    # 2个值的类型都是pytorch tensor(pt)，分别表示 prompt对应的token id 及 每个id的有效性


# 尝试把 tokens 拼成字符串
text_reconstructed = tokenizer.decode(tokens["input_ids"][0][0], skip_special_tokens=True)
# print("Reconstructed text:", text_reconstructed)

# 使用special token占位图像，明确告诉模型图像信息的位置
text_with_image = "这里有一张图片 <|im_start|> ... <|im_end|>"
tokens_with_image = tokenizer(text_with_image, return_tensors="pt")
# print("Tokens with <|im_start|>:", tokens_with_image)


# image_processor
image_processor = processor.image_processor

from PIL import Image
image = Image.open("/home/lenovo/Pictures/801.jpg").convert("RGB")  # 转为 RGB

# image_processor 预处理
image_inputs = image_processor(images=image, return_tensors="pt") 
    # 输出一个字典 其中包含2个键 "pixel_values", "image_grid_thw"
    # 分别表示预处理后的图像tensor，喂给视觉塔； 图像划分的grid 维度为(batch, 高度方向的patch(行), 宽度方向的patch（高）)；


# Qwen2.5-VL-3B-Instruct 模型中图像预处理参数 min_pixels 和 max_pixels 的说明，
# 具体用于控制每张图像被切分成多少视觉 token（视觉 patch），从而影响模型的输入大小、计算量以及推理成本
# 设置较低的 token 数 → 更快的推理，但可能损失细节
# 设置较高的 token 数 → 更高的精度，但计算成本更高
# 视觉 token 是将图像划分为小块（如 28x28 的 patch）后，送入模型处理的最小视觉单元
# 每张图像最多会被处理成 16384 个视觉 token（最少为 4 个）
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28 (每张图处理为 256个token)
# max_pixels = 1280*28*28 (每张图处理为 1280个token)
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# 1. 图像理解
from PIL import Image
import requests

from pathlib import Path
#local_image_path = Path("/home/lenovo/Pictures/801.jpg")  # 替换为你本地图片路径
#local_image_path = '/home/lenovo/Downloads/test_vlm/clothes.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/flip_package1.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/flip_package2.jpg'
local_image_path = '/home/lenovo/Downloads/test_vlm/scan_medicine1.jpg'
image = Image.open(local_image_path).convert("RGB")

# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
# image.show()

prompts = [
        '描述图像中两个机器人手臂的位置和姿态。它们目前处于什么状态',
        '如果指令是“从左侧药箱中拣选一个药盒，扫描其条形码后放入右侧入库箱”，你会如何理解并执行这个任务',
        '描述图像中两个机器人手臂的位置和姿态。它们目前处于什么状态？',
        '描述图像中包裹的放置情况',
        '你是人形机器人,你的任务是面前所有包裹二维码朝上滑到你右侧的传送带上，基于当前图像展现的情况，你会先处理哪个包裹？',
        '你是人形机器人，如果接到指令“将所有包裹的二维码标签翻转朝上，然后挪到右侧传送带上“，基于当前图像展现的情况，你会如何操作？',
        '描述图像左侧包裹的颜色和二维码标签的状态（是否朝上）。',
        '详细描述图像中所呈现的内容。',
        '假设你是人形机器人，图像你所看到的画面，你的任务是将衣服叠好，你将怎样操作？'
        ]

# 多模态对话格式
messages = [
    {
        "role": "user",  # role：发送者身份，user / assistant 等
        "content": [  # 消息内容列表，可同时包含 图像和文本
            {
                "type": "image",
                "image": image, #"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            # {"type": "text", "text": prompts[0]}
            {"type": "text", "text": "I have an item to place. Based on this pictrue, what is the appropriate way to navigate to the shelf?"},
        ],
    }
]

# 把messages转换成模型期望的 统一多模态对话格式
    # 将多模态消息（图像+文本） 按模型要求加上special token，形成标准对话格式。tokenize=False 表示messages不会转化为token id
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, system_prompt="你是一个聊天机器人"
)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>

# <|im_start|>user
# <|vision_start|><|image_pad|><|vision_end|>I have an item to place. Based on this pictrue, what is the appropriate way to navigate to the shelf?<|im_end|>

# <|im_start|>assistant

# 从 messages 中提取 图像/视频数据
image_inputs, video_inputs = process_vision_info(messages)

# 文本tokenization：将 text 转为 input_ids 和 attention_mask
# 图像/视频处理：把 image_inputs 和 video_inputs 对齐到模型输入结构 （没过siglip）
# padding：对文本序列做 padding，保证 batch 内长度一致

# inputs是一个字典 其中包含 "input_ids"、"attention_mask"、"pixel_values"、"image_grid_thw"等键
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,  # 对文本长度做统一处理
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 只提供 语言embedding 后的结果 的接口
# print(inputs["input_ids"].shape)
text_embedding = model.model(inputs["input_ids"])
# print(text_embedding.last_hidden_state.shape)

# 推理生成  最多 256 个新 token
# **inputs 会把字典里的 key 当作参数名，把 value 当作参数值传给函数
generated_ids = model.generate(**inputs, max_new_tokens=256)

# 直接解码整个 token 序列，包括输入
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,  # 可保留特殊符号，也可以设置 False
    clean_up_tokenization_spaces=False
)

# 输出模型生成的文字，包括输入
# print(output_text)

# # 去掉输入部分的 token，只保留模型生成的部分
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]

# # 解码token id为文本字符串，生成最终的图像描述结果
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )

# # 输出模型生成的文字
# print(output_text)

# 2. 多张图像理解

# 3. 视频理解
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "/home/lenovo/Videos/shelves/001.jpg",
                    "/home/lenovo/Videos/shelves/002.jpg",
                    "/home/lenovo/Videos/shelves/003.jpg",
                    "/home/lenovo/Videos/shelves/004.jpg",
                    "/home/lenovo/Videos/shelves/005.jpg",
                    "/home/lenovo/Videos/shelves/006.jpg",
                    "/home/lenovo/Videos/shelves/007.jpg",
                    "/home/lenovo/Videos/shelves/008.jpg",
                    "/home/lenovo/Videos/shelves/009.jpg",
                    "/home/lenovo/Videos/shelves/010.jpg",
                    "/home/lenovo/Videos/shelves/011.jpg",
                ],
            },
            {"type": "text", "text": "这段视频是一个人轨迹观测，根据这段视频，判断他这是想去哪里?"},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    # fps=30,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
# 打印这个inputs['input_ids']
print(inputs['input_ids'].shape)  # inputs['input_ids'] 二维张量 [batch_size, seq_length]
print(len(inputs['input_ids']))  # batch_size
print(inputs['pixel_values_videos'].shape)

print(tokenizer.decode(inputs['input_ids'][0]))  # 看到真实的 prompt 内容（带 <|video_start|>、<|video_pad|>、<|video_end|> 等）
# <|video_pad|> 只是一个占位符，用于在 token 序列中标记视频位置。
# 真正的视频信息在 inputs['pixel_values_videos'] 里，由模型在 forward 时融合进文本输入。


inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=512)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
