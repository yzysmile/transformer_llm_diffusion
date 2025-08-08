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
local_image_path = Path("/home/lenovo/Pictures/801.jpg")  # 替换为你本地图片路径
image = Image.open(local_image_path).convert("RGB")

# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
# image.show()

# 多模态对话格式
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image, #"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "I have an item to place. Based on this pictrue, what is the appropriate way to navigate to the shelf?"},
        ],
    }
]

# 构造用于推理的输入（包含图像和文本）
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 推理生成  最多 128 个新 token
generated_ids = model.generate(**inputs, max_new_tokens=256)

# 去掉输入部分的 token，只保留模型生成的部分
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 解码为文本字符串，生成最终的图像描述结果
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 输出模型生成的文字
# print(output_text)

# 2. 多张图像理解




# 3. 视频理解
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/home/lenovo/Pictures/801.mp4",
                "max_pixels": 960 * 540,
                "fps":5.0,
            },
            {"type": "text", "text": "这是一段人行走时的观测，他是在朝着架子走去吗？"},
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
