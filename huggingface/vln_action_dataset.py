import os
import torch
import json
import copy
import random
import tokenizers
import numpy as np
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, Optional, Sequence, List
from PIL import Image
from packaging import version

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import *

from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
from streamvln.args import DataArguments

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    """功能：为对话添加角色标识符和信号标记"""
    # header 系统提示
    # source 原始对话列表（每轮包含"from"和"value"）
    # get_conversation是否返回完整拼接的对话

    # # 输入
    # header = "System: You are an assistant." 
    # source = [{"from":"human", "value":"Hello"}, {"from":"gpt", "value":"Hi"}]  

    # # 输出
    # "System: You are an assistant.### Human: Hello\n### GPT: Hi\n### "
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """功能：处理含图像标记的多模态对话"""
    # 检测对话中的图像占位符 DEFAULT_IMAGE_TOKEN
    # 规范化图像标记位置（确保出现在句首）
    # 根据配置添加图像起止标记（<im_start>和<im_end>）
    # 清理噪声数据（如"QA_GT_caption_based_noisy"）
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources: # 遍历每个对话样本
        for sentence in source: # 遍历每轮对话
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            # 图像标记规范化
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"])) # 计算当前句子中图像标记 <image> 的数量
    
            # 当句子中有且只有一个图像标记，且标记不在句首时，确保图像标记出现在文本开头（符合LLM视觉处理惯例）
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()  # 删除句子中现有的图像标记，.strip() 移除首尾空白字符
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]  # 重建句子：在句首添加图像标记 + 换行符
                sentence["value"] = sentence["value"].strip() # 再次清理首尾空白（确保格式规范）
                if "mmtag" in conversation_lib.default_conversation.version:  # 如果对话版本是 "mmtag"
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
                    # 包裹图像标记：<image> → <Image><image></Image>
            
            # 添加图像起止标记 <im_start><image><im_end>
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end: # 条件检查：如果配置要求使用图像起止标记
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            # 噪声清理（来源：videoInstruct-100k 数据集中的噪声）
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources
# e.g 输入
# [
#     {
#         "from": "human",
#         "value": "What's in this image? <image>"
#     }
# ]

# e.g 输出
# [
#     {
#         "from": "human",
#         "value": "<im_start><image><im_end>\nWhat's in this image?"
#     }
# ]



# LLaMA-2 对话模板特点
# 特殊标记：
# [INST]：用户输入开始
# [/INST]：用户输入结束/助手回复开始
# <s>/</s>：序列开始/结束

# 系统提示：
# text
# <s>[INST] <<SYS>>系统提示<</SYS>>用户问题 [/INST]

# 多轮结构：
# text
# <s>[INST] 第一轮问题 [/INST] 第一轮回答 </s>
# <s>[INST] 第二轮问题 [/INST] 第二轮回答 </s>

# 特定模型的预处理函数
# 专门为LLaMA-2模型设计的对话数据预处理函数，主要完成三个核心任务：
    # 按照 LLaMA-2 的特殊格式组织对话
    # 分词处理：将文本转换为token ID序列
    # 标签掩码：标记需要计算loss的部分

# sources = [
#     [
#         {"from": "human", "value": "What is the capital of France?"},
#         {"from": "gpt", "value": "The capital of France is Paris."}
#     ],
#     [
#         {"from": "human", "value": "Explain photosynthesis."},
#         {"from": "gpt", "value": "Photosynthesis is the process..."}
#     ]
# ]

# <<SYS>>
# You are a helpful assistant.
# <</SYS>>

# [INST] What is the capital of France? [/INST] The capital of France is Paris.</s>

# [INST] What is the capital of France? [/INST] The capital of France is Paris.</s>
def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    # 输入
        # sources:原始对话数据
        # tokenizer:分词器
        # has_image:是否包含图像

    # 返回字典: {input_ids, labels}
    """
    # 1. 对话模板初始化
    conv = conversation_lib.default_conversation.copy()  # 复制默认对话模板（包含角色定义、分隔符等）
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # 创建角色映射字典（"human"→"Human"，gpt→"Assistant"）

    # 2. 应用对话模板
    conversations = []
    for i, source in enumerate(sources):
        # 确保对话以用户开始
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        
        # 按轮次构建对话
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 验证角色交替顺序（用户→助手→用户...）
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        
        # 生成格式化对话字符串
        conversations.append(conv.get_prompt())

    # 3. 分词处理
    if has_image: # 多模态分词（处理图像标记）
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:  # 纯文本分词
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",  # 按批次最长序列填充
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # 4. 创建标签掩码
    targets = input_ids.clone()
        # 验证分隔符风格是LLaMA-2
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "  # LLaMA-2的用户/助手分隔符
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 按对话轮次分割
        rounds = conversation.split(conv.sep2) # sep2通常是"</s>"
        cur_len = 1 # 跳过BOS token
        target[:cur_len] = IGNORE_INDEX # 忽略BOS token

        # 处理每轮对话
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            
            # 分割用户输入和助手回复
            parts = rou.split(sep)
            if len(parts) != 2:
                break  # 无效格式跳过
            parts[0] += sep # 重建完整用户输入

            if has_image: # 计算长度（考虑多模态）
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # 掩码用户输入部分（不计算loss）
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len # 移动到下一轮

        # 掩码填充部分
        target[cur_len:] = IGNORE_INDEX

        # 长度校验
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    # stop_token_index = tokenizer.convert_tokens_to_ids("Ġstop")

    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX
            
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

class VLNActionDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        data_args, 
        task_id
    ):
        super(VLNActionDataset, self).__init__()

        # 读取配置与参数
        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        self.video_folder = data_args.video_folder.split(',')

        # 加载数据文件和注释
        self.nav_data =[]
        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        # 构造训练样本 self.data_list
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]
                
            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)
                    if valid_idx != 0:
                        invalid_len += 1

                if actions_len - valid_idx < 4:
                    continue
                
                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        # 定义动作映射与 prompt 模板
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }
        # 动作使用四种离散符号表示，并用于训练时生成目标文本序列。

        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.act_conjunctions = [
                                    'and then ', 
                                    'after that ', 
                                    'next ', 
                                    'the next action is ',
                                    'followed by ', 
                                    'leading to ', 
                                    'continuing ',
                                    'subsequently ', 
                                    'proceeding to '
                                ]
        
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

    def __len__(self):
        return len(self.data_list)
    
    @property
    def task(self):
        return self.task_id
    
    # 将动作序列变成字符串
    def actions2text(self, actions):
        converted_sequence = []         
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        
        text = ''.join(converted_sequence)
        return text
    
    # 构造语言提示（含图像 token）
    def prepare_conversation(self, conversation, actions): 
        i = 0
        sources = []
        t = 0
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            source[1]["value"] = answer
            i += len(step_actions)
            t += 1
            sources.extend(source)
        return sources
    
    # 样本生成主函数
    def __getitem__(self, i):
        # 1. 数据索引解析与加载
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        # ep_id: 当前 episode 的 ID（一个导航轨迹）
        # ins_id: 当前使用哪条指令
        # start_idx: 当前样本的起始时间戳
        # valid_idx: 当前视频中第一个有效帧（跳过前面可能无动作的帧）
        data = self.nav_data[ep_id]

        video_path = data['video']  # 当前 episode 的 RGB 视频帧路径
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))  # 所有帧文件名，按顺序排列

        # 2. 获取文本指令
        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        # 3. 动作裁剪与时间索引计算
        actions = data['actions'][1+valid_idx:] + [0]
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        actions = np.array(actions)[time_ids]

        # 4. 样本帧选取（未来轨迹）
        start_idx, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
        
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]

        # 5. 历史帧选取（记忆机制）
        if time_ids[0] != 0:
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]
        else:
            history_frames = []
        
        # 6. 图像读取与变换
        images = []
        for image_file in sample_frames + history_frames:
            image = Image.open(image_file).convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0] # [3, H, W]
            images.append(image)

        images = torch.stack(images)
        
        # 7. 构造语言输入文本（对话格式）
        sources = copy.deepcopy(self.conversations)

        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
        
        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instructions[ins_id])
        interleave_sources = self.prepare_conversation(sources, list(actions))
        
        # 8. 文本 tokenization + 模型输入准备
        data_dict = preprocess([interleave_sources], self.tokenizer, True)

        # 9. 最终返回样本内容
        return data_dict["input_ids"][0], \
            data_dict["labels"][0], \
            images, \
            torch.tensor(time_ids), \
            self.task

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

# 用于 DataLoader 的 batch 合并逻辑
def collate_fn(batch, tokenizer):
    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*batch)
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=IGNORE_INDEX)
    
    input_ids_batch = input_ids_batch[:, :tokenizer.model_max_length]
    labels_batch = labels_batch[:, :tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)
    
    img_lens = np.array([i.size(0) for i in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(time_ids_batch, batch_first=True, padding_value=-1)
    
    image_batch = pad_tensors(image_batch, img_lens)
    
    return {'images': image_batch, 
            'time_ids': time_ids_batch, 
            'attention_mask': attention_mask, 
            'input_ids': input_ids_batch, 
            'labels': labels_batch, 
            'task_type': task_type_batch}