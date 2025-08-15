import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL3-1B"
#model = AutoModel.from_pretrained(
#    path,
#    torch_dtype=torch.bfloat16,
#    low_cpu_mem_usage=True,
#    use_flash_attn=True,
#    trust_remote_code=True).eval().cuda()

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-1B'
#local_image_path = Path("/home/lenovo/Pictures/801.jpg")  # 替换为你本地图片路径
local_image_path = '/home/lenovo/Downloads/test_vlm/clothes.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/flip_package1.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/flip_package2.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/scan_medicine1.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/scan_medicine2.jpg'
#local_image_path = '/home/lenovo/Downloads/test_vlm/scan_QR2.jpg'
prompts = [
        '任务指令是将包裹二维码朝上放置到扫码台扫码，然后放到传送带上。基于这张图，你应该处理哪个>包裹，如何处理',
        '包裹、扫码区和传送带的具体位置和特征是什么？它们是如何排列的',
        '指令是“将所有包裹的二维码方向朝上进行扫码，然后放到右侧传送带上”，你会如何理解并执行这个>任务',
        '任务是药盒的拣选、扫码、入库，当前机器人应该执行扫码操作',
        '描述机器人手中的药盒状态',
        '描述图像中两个机器人手臂的位置和姿态。它们目前处于什么状态',
        '如果指令是“从左侧药箱中拣选一个药盒，扫描其条形码后放入右侧入库箱”，你会如何理解并执行这>个任务',
        '描述图像中两个机器人手臂的位置和姿态。它们目前处于什么状态？',
        '描述图像中包裹的放置情况',
        '你是人形机器人,你的任务是面前所有包裹二维码朝上滑到你右侧的传送带上，基于当前图像展现的情况，你会先处理哪个包裹？',
        '你是人形机器人，如果接到指令“将所有包裹的二维码标签翻转朝上，然后挪到右侧传送带上“，基于>当前图像展现的情况，你会如何操作？',
        '描述图像左侧包裹的颜色和二维码标签的状态（是否朝上）。',
        '描述图像中所呈现的内容。',
        '假设你是人形机器人，图像你所看到的画面，你的任务是将衣服叠好，你将怎样操作？'
        ]
image = load_image(local_image_path)
#image.show()
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe((prompts[-2], image))
print(response.text)
