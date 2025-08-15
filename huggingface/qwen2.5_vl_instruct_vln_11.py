from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import json
import os
import numpy as np
from PIL import Image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


from typing import List
# 要把数据处理成 与 qwen2.5-vl-3b-instruct
def qwen_format_prompt(video: List[str]):
    # video是一个图像路径列表
    # system信息
    # 假设你是机器人,正在执行导航任务

    messages = [
        {
            "role": "system",
            "content": "你是一个机器人，正在执行导航任务。"
        },
        {   
            "role": "user", 
            "content": [ 
                { 
                    "type": "video", 
                    "video": video, 
                },
                {
                    "type": "text", 
                    "text": "你现在的任务是找到视野中的货架后移动到货架旁，若视野中没有货架，先原地旋转直到货架出现在视野中央，再移动到货架旁边后停止。请根据这段观测视频来决定你接下来的动作，接下来的动作只能是：左转、右转、前进、停止、任务完成 或其组合"
                }, 
            ], 
        }
     ]
    
    return messages

def preprocess(video: List[str], actions_label_text: str):
    """为不同的backbone配置"""
    return preprocess_qwen(video, actions_label_text)

def preprocess_qwen(video, actions_label_text):
    messages = qwen_format_prompt(video)
    messages.append({"role": "assistant", "content": actions_label_text})
    return messages

def find_assistant_tokens(tokenizer, input_ids):
    """
    找出 <|im_start|>assistant ... <|im_end|> 区间
    返回这些区间的 起始和结束索引，用于后续生成 labels 只计算 assistant 回复部分的 loss
    
    args: 
        tokenizer: 
        input_ids: 一维张量
    """
    result = []
    start_index = 0
    assistant_id = tokenizer("assistant")["input_ids"][0]
    im_end_id = tokenizer("<|im_end|>")["input_ids"][0]

    while start_index < len(input_ids):
        if input_ids[start_index] != assistant_id:
            start_index += 1
        else:
            # 找到 assistant 开始
            end_index = start_index + 1
            while end_index < len(input_ids) and input_ids[end_index] != im_end_id:
                end_index += 1
            # 加入区间（可选择是否包含 start/end token）
            result.append((start_index + 1, end_index))  # 跳过 <|im_start|>assistant
            start_index = end_index + 1
    return result

from torch.utils.data import Dataset
class B4Dataset(Dataset):
    def __init__(self, processor, data_args):
        super().__init__()
        self.processor = processor  # 对于VLM，其中包含tokenizer、image_processor、video_processor

        self.num_history = data_args.num_history  # 历史观测
        self.num_frames = data_args.num_frames # 当前观测（由最近的num_frames张图像构成）
        self.num_future_steps = data_args.num_future_steps  # 每次预测时考虑的未来动作步数
        self.video_folder = data_args.video_folder.split(',') # 视频数据集路径，可以支持多个路径（用逗号分隔）

        # 加载各数据集的annotations.json文件 变为列表后 混合在一起后变为self.nav_data一个大列表
        self.nav_data = []

        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))  # 列表
            for tdata in anno_json:  # anno_json文件是一个列表（list），列表中的每个元素是一个字典
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        # 构造训练样本是由一个个完整导航组成拆分成的片段
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            # ep_id是 0, 1, 2, 3, 4 ... n
            # item是一个 字典
            actions = item['actions'] # 列表
            actions_len = len(actions)
            if actions_len < 4:  # 这是因为默认的是要预测未来4步动作吗???
                continue

            num_rounds = actions // self.num_frames  # 把剩余动作按self.num_frames作为步长划分得到整除次数 向下取整。
            for n in range(num_rounds + 1):
                if n * self.num_frames == actions_len:
                    continue
                self.data_list.append((ep_id, n * self.num_frames)) 
                # e.g：
                # actions_len = 23
                # self.num_frames = 8
                # 循环 n=0,1,2
                # 检查：
                # n=0: 起点=0 ≠ 23 → 保留
                # n=1: 起点=8 ≠ 23 → 保留
                # n=2: 起点=16 ≠ 23 → 保留
                # 最终：
                # (ep_id, 0)  
                # (ep_id, 8)
                # (ep_id, 16) 
                
                # 把每一条导航 拆分成一条条片段，这里还没有正儿八经拆成片段，只是明确要怎么拆
        
        self.idx2actions = {
            '0': "STOP",
            '1': "↑",
            '2': "←",
            '3': "→",
        }

    def actions2text(self, actions):
        "动作真值(label)列表 成为一个连续的字符串"
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions[str(action)]
            converted_sequence.append(act_text)
        
        text = ''.join(converted_sequence)
        return text
       
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        ep_id, start_idx = self.data_list[i]
        # ep_id——所有数据集中的某一条完整导航的索引
        # tart_idx——从ep_id这一条完整导航的那一步开始拆分

        # 索引为ep_id的完整导航数据
        data = self.nav_data[ep_id]
        video_path =  data['video'] # 索引为ep_id完整导航数据的对应观测路径
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb'))) # 完整的一条导航数据的对应观测

        actions = data['actions'][1:] + [0] # 末尾加上一个 0 表示在动作序列最后补一个 STOP（动作索引为0），方便模型学习序列终止
        actions_len = len(actions)  # 计算当前有效动作序列的长度（包括最后补的 STOP）

        # 将ep_id这一条完整的轨迹 切成导航片段 最终获得 该导航片段对应观测图像 和 要预测的动作标签
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))  # [start_idx, start_idx + num_frames)，最多取num_frames帧（如果actions_len不够，就取到最后一个动作）
        assert len(time_ids) > 0
       
        start_idx, end_idx, interval = time_ids[0], time_ids[-1]+1, self.num_future_steps  # x[start:end)是左闭右开，end索引要比最后一个元素大1才能完整取到它
        sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]  # 作为当前观测

        if time_ids[0] != 0: # 若该片段的起始帧不是0，则之前片段对应的观测可当做 历史观测
            history_step_ids = np.arange(
                0,
                time_ids[0],
                max(time_ids[0] // self.num_history, 1) # 采样步长，均匀采样self.num_history张历史帧
            )

            # 根据采样得到的历史帧时间索引，拼接成完整的帧文件路径列表
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]

        else:
            history_frames = []

        # 该导航片段对应所有观测
        images = []
        for image_file in (sample_frames + history_frames):
            images.append(image_file)

        # 该导航片段观测要预测的未来动作真值
            # # 动作预测的真值不用interval 以保证动作序列连续性
            # actions = np.array(actions)[end_idx: min(end_idx + self.num_future_steps, actions_len)]  

            # 动作预测的真值改成带interval的版本
        actions = list(
            np.array(actions)[
                np.arange(
                    end_idx,
                    min(end_idx + self.num_future_steps * interval, actions_len),
                    interval
                )
            ]
        )

        actions_label_text = self.actions2text(actions)

        # 为backbone模型输入准备
        messages = preprocess(images, actions_label_text)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, system_prompt=None
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = inputs['input_ids'] # inputs['input_ids']是二维张量 torch.Size([1, xxx])
        indexs = find_assistant_tokens(self.processor.tokenizer, input_ids[0])

        labels = len(input_ids[0]) * [self.processor.tokenizer.pad_token_id] # 初始化label 且全为pad
        labels = torch.tensor([labels], dtype=torch.long)  # 把labels转化为 ([1, xxx])

        # 遍历每个 assistant 内容区间，把对应位置的label填成真实token id，其它地方依然是pad
        for index in indexs:
            labels[index[0]:index[1]] = input_ids[0][index[0]:index[1]]

        input_ids = input_ids[:-1]
        labels = labels[1:]

        pixel_values = inputs['pixel_values'] # 二维张量 ([总patch数, 每个patch的embedding 维度])；总patch数=num_frames × patches_per_frame
        
        return {
            'input_ids': input_ids, # 二维张量 ([1, xxx])，实际有效的是 "xxx"
            'labels': labels, # 二维张量 ([1, xxx])
            'pixel_values': pixel_values # 二维张量
        }
    
# input_ids labels pixel_values长度可变, 直接用DataLoader默认的batch_size会报错；pixel_values 的 patch 数可能不同 不能直接堆叠成一个torch.Tensor
# 写一个专门的collate_fn来给 DataLoader 使用
import torch
def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """把长度不一的tensor(如视频 patch[num_patches, dim])padding 到相同长度
    
    args:
        tensors: 是一个list, 每个元素是一个tensor
        lens: 每个tensor的长度 
        max_len: padding 的目标长度, 默认取tensors中最长的tensor
        pad: padding值
    """
    if lens is None:
        lens = [t.size(0) for t in tensors]  # t.size(0)中的0 是不是要改？
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors) # bs——batch size，列表tensors中有多少个张量
    hid = tensors[0].shape[1:]  # hid 是剩余维度 取第一个张量 tensors[0] 的形状，忽略第 0 维（序列/帧长度）
    dtype = tensors[0].dtype # 取第一个张量的数据类型
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    # 初始化输出张量 output
    # 形状为 [batch_size, max_len, ...hid]
    # 第0维：batch
    # 第1维：pad 后的序列长度
    # 其余维度：保持原始张量维度
    # to(tensors[0].device) 保证在同一个设备（CPU 或 GPU）上

    if pad:
        output.data.fill_(pad)  # 如果传入 pad 值（通常非 0），就把 output 的所有元素填成 pad

    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    # 遍历每个输入张量 t 以及对应长度 l
    # 将 t 的内容复制到 output 对应位置
    # :l 表示只填充原始张量长度的部分，其余部分保持 pad

    return output
    # 返回 padding 后的张量，形状 [batch_size, max_len, ...hid]
    # 这样就可以直接用在模型输入或者 DataLoader batch 中
                    




        
        
