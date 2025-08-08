import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

# 导入本地图像
from pathlib import Path
local_image_path = Path("/home/lenovo/Pictures/801.jpg")  # 替换为你本地图片路径
image = Image.open(local_image_path).convert("RGB")
image.show()

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    # 用 Processor 预处理输入
    # 输入为文本（任务 prompt）+ 图像，processor 会：
    # 将 prompt tokenize 成 input_ids
    # 将图像转换为模型可接受的 pixel_values
    # 输出 PyTorch 格式的张量 (return_tensors="pt")
    # 放到指定设备上（GPU/CPU）
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # 生成模型输出
        # 用 .generate() 进行文本生成（VLM 模型的输出是文本格式的描述）。
        # input_ids 提供 prompt，pixel_values 提供图像信息。
        # max_new_tokens=4096: 最多生成 4096 个 token。
        # num_beams=3: 使用 beam search 解码，提升质量。
        # do_sample=False: 不随机采样，确保稳定输出。
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )

    # 解码生成的文本
    # 将生成的 token ids 解码为文本字符串
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 后处理成结构化检测结果
    # 这是 Florence-2 的关键处理：
    # 它会把生成的文本（如 [{"label": "car", "box": [x1, y1, x2, y2]}]）转换为结构化的结果（字典或列表），并根据图片尺寸进行缩放。
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    print(parsed_answer)


prompt = "<CAPTION>"
run_example(prompt)

prompt = "<DETAILED_CAPTION>"
run_example(prompt)

prompt = "<MORE_DETAILED_CAPTION>"
run_example(prompt)

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")

prompt = "<OD>"
run_example(prompt)

prompt = "<DENSE_REGION_CAPTION>"
run_example(prompt)

prompt = "<REGION_PROPOSAL>"
run_example(prompt)

prompt = "<OCR>"
run_example(prompt)

prompt = "<OCR_WITH_REGION>"
run_example(prompt)


def run_example_with_score(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      return_dict_in_generate=True,
      output_scores=True,
    )
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]

    prediction, scores, beam_indices = generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices
    transition_beam_scores = model.compute_transition_scores(
        sequences=prediction,
        scores=scores,
        beam_indices=beam_indices,
    )

    parsed_answer = processor.post_process_generation(sequence=generated_ids.sequences[0], 
        transition_beam_score=transition_beam_scores[0],
        task=task_prompt, image_size=(image.width, image.height)
    )

    print(parsed_answer)

prompt = "<OD>"
run_example_with_score(prompt)