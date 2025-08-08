# Qwen2.5-0.5B-Instruct(Base language model) 和 SigLip-400M
# 多模态大模型最重要的一环就是 文本-图像 对齐到 同一空间


from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers import PretrainedConfig, AutoConfig

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    # 在 VLMConfig 的PretrainedConfig子类中 通过 自定义字符串 定义模型类型。
    # model_type 只是一个 标识符，用来匹配 AutoConfig 和 AutoModelForCausalLM 的注册表
    # 若没有用 AutoConfig.from_pretrained() 或 AutoModelForCausalLM.from_pretrained() 直接加载 VLMConfig，那么可以删除这一行，不会影响模型的正常运行。

    def __init__(self, llm_model_path='/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                 vision_model_path='/home/user/Downloads/siglip-so400m-patch14-384',  # 384x384的图像—> 14*14的patch
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

# 注册配置到 AutoConfig， 现在huggingface的AUTO类 有了VLMConfig
AutoConfig.register("vlm_model", VLMConfig)


class VLM(PreTrainedModel):
    config_class = VLMConfig # 绑定VLM类的config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 模型（Backbone）
        # AutoModel——Transformer 主干（用于特征提取）
        # AutoModel加载的模型只能接受已经预处理好的输入（比如 token ids、图像张量），不负责原始数据的转换

        # AutoModelForCausalLM——自回归语言模型（如 GPT/LLaMA）
        # AutoModelForMaskedLM——掩码语言模型（如 BERT）
        # AutoModelForSeq2SeqLM——序列到序列任务（如 T5、BART 用于翻译、摘要）

        # Tokenizer
        # AutoTokenizer——用于加载 对应模型的分词器（Tokenizer），可用于文本预处理，将输入文本转换为 input_ids(token)

        # 特征提取
        # AutoProcessor——适用于 图像+文本、语音+文本等多模态任务，同时包含 Tokenizer 和 Feature Extractor。不需要手动去找对应的 tokenizer、image processor 等类，直接调用 AutoProcessor 就能自动匹配
        # 但不包含大模型本身
        # 如果只想用纯文本tokenizer，也可以直接用AutoTokenizer，但多模态模型就用 AutoProcessor 更方便。

        # AutoImageProcessor——专门用于计算机视觉任务，如 ViT、DETR

        # 模型配置
        # AutoConfig—— 用于获取模型的超参数（hidden_size、num_layers 等），无需加载整个模型。
        # 没有使用 AutoConfig，是因为 config 已经由 VLMConfig 直接提供了模型的必要超参数，无需再调用 AutoConfig。

        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        # 这会加载 siglip-so400m-patch14-384 的模型权重和结构，如 SiglipVisionModel 或 CLIPVisionModel，具体取决于模型的类型

        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        # 会加载 siglip-so400m-patch14-384 对应的处理器，负责调整大小、归一化等，使输入图片适配模型。

        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)

        # 视觉模型 self.vision_model 提取的图像特征维度 与 语言模型 llm_model 的嵌入维度不同，
        # 故 需要 Linear 层来 对齐特征维度
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size * 4,
                                 self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False # 冻结 vision_model，避免调整其权重

        for param in self.llm_model.parameters():
            param.requires_grad = False # self.llm_model 也 全部冻结，防止语言模型的参数更新。

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # input_ids 是经过对文本 tokenizer得到   pixel_values 经过对图像 预处理得到
        # 两者在加载数据MyDataset时得到

        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        # 通过 llm_model 的内部接口“.get_input_embeddings”得到token的embedding
        # 获取 LLM 的词向量嵌入层，将 input_ids(token) 转换为 text_embeds，即 (batch, seq_len, hidden_dim)

        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        # pixel_values 是 一张或多张经过预处理(e.g. 尺寸调整、图像中心裁剪、归一化、转换为张量、数据增强...)的图像数据，
        # 通过 vision_model(siglip、CLIP 或 ViT 等) 提取特征，得到 image_embeds

        #  压缩图片 Token 数
        b, s, d = image_embeds.shape
        # b- b张图像
        # s— 每张图像被拆分成的 patch 数量 14*14=196
        # d- 每个patch的特征向量维度
        image_embeds = image_embeds.view(b, -1, d * 4)  # e.g (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.linear2(F.silu(self.linear1(image_embeds))) # (b, 49, d)

        text_embeds = text_embeds.to(image_features.dtype) # 使text_embeds 和 image_features 具有相同的数据类型

        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        # image_features 插入 text_embeds 的合适位置，形成 最终的输入嵌入 inputs_embeds。

        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) # 前向传播
        # 直接使用 inputs_embeds 而不是 input_ids，inputs_embeds输入已经包含视觉信息

        logits = outputs[0]
        # logits 是模型的原始输出，通常是一个形状为 (batch_size, seq_length, vocab_size) 的张量，
        # vocab_size表示下一个 token 的预测分布

        loss = None
        if labels is not None: # 如果 labels 不是 None，说明是在训练/微调，就计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            # 忽略 PAD token，不影响损失计算
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),  # 三维 转化为 二维，“-1”表示维度自己填充、logits.size(-1)表示vocab_size
                # logits：(batch_size, sequence_length, vocab_size) 转换为 (batch_size * sequence_length, vocab_size)
                # 将所有的 token 展平成一个长向量，便于与 labels 进行比较

                labels.view(-1).to(logits.device)
                # (batch_size, sequence_length) 转换为 (batch_size * sequence_length,)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        # 返回 CausalLMOutputWithPast 对象，它是 transformers 库中的一个标准输出格式，用于封装模型的输出
        # 包含两个主要字段 loss和logits

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # self.tokenizer('<|image_pad|>')['input_ids'][0] 是一个特殊token <|image_pad|> 的 ID，该 token 用于标记图像部分的输入;
        # torch.where(input_ids == ...) 返回了 input_ids(token) 中所有与 <|image_pad|> 相等的位置的索引。
        # batch_indices——这些位置的批次索引； image_indices：这些位置的 token 在该批次中的位置索引

        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim) # (b, 49, d) -> (b*49, d)

        return inputs_embeds

# 告诉 AutoModel，当它遇到 VLMConfig 这个配置类时，应该使用 VLM 这个模型类。
AutoModel.register(VLMConfig, VLM)

class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        # 第index个sample的样子，是一个dict
        # {
        #     "image": "example.jpg",
        #     "conversations": [
        #         {"role": "user", "value": "这张图片里有什么？"},
        #         {"role": "assistant", "value": "图片中有一只猫"}
        #     ]
        # }

        try:
            image_name = sample['image']
            conversations = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": conversations[0]['value']}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            # 使用apply_chat_template 使得输入符合模型的 对话格式（包含 system prompt + user prompt）
            # 构造 q_text（问题文本），分成两个部分来看：
            # 使用 apply_chat_template() 生成标准的对话格式；
            # 用户输入包含 <image> 占位符，它会被填充 image_pad_num 个 <|image_pad|>

            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            # 构造 a_text（答案文本），a_text 只是 模型应该生成的文本，a_text 只需要是纯文本答案，并以 eos_token 结尾，以便模型知道何时停止。

            q_input_ids = self.tokenizer(q_text)['input_ids']
            # tokenizer(q_text)['input_ids']将q_text转换成Token ID列表 q_input_ids（token）

            a_input_ids = self.tokenizer(a_text)['input_ids']
            # tokenizer(a_text)['input_ids']处理a_text，得到 a_input_ids(token)。

            input_ids = q_input_ids + a_input_ids
            # input_ids = q_input_ids + a_input_ids：将问题(q_input_ids)和答案(a_input_ids)拼接，形成完整的输入序列。

            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            # q_input_ids 这部分的 labels 设为 pad_token_id（不计算 loss）；
            # a_input_ids 这部分保留，作为 训练目标。

            input_ids = input_ids[:-1]
            labels = labels[1:]
            # 训练时，模型应该预测下一个 token，
            # 让 labels[t] 变成 input_ids[t+1] 的监督信号

            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            # 读取 image_name 并转换为 RGB 格式（确保 3 通道）。

            pixel_values = self.processor(text=None, images=image)['pixel_values']
            # 视觉处理器processor负责 归一化 + 预处理。
            # pixel_values是标准化后的张量，可直接输入视觉模型。
        except: #  处理异常情况
            default_image = Image.new('RGB', (224, 224), color='white')
            # 如果读取图像失败，创建白色填充图像 224×224 以避免训练时出错

            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": "图片内容是什么\n<image>"}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            # 用户输入改为 "图片内容是什么"。
            # AI 回答 "图片内容为空"（表示无效图像）。

            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }
        # 返回单个数据样本 字典结果
        # input_ids：模型的文本输入（问题 + 答案）；
        # labels：训练目标，q_input_ids 处填充 pad_token_id，a_input_ids 处保留；
        # pixel_values：预处理后的图像张量

class MyDataCollator:
    # 主要用于 批处理（batching），它的作用是：
    # 对文本数据进行填充（Padding），确保 input_ids 和 labels 具有相同的长度。
    # 对图像数据进行堆叠（Concatenation），确保 pixel_values 以合适的格式输入到模型中
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    # __call__ 方法会在 DataLoader 取数据时自动调用，把 MyDataset 生成的多个样本 合并成一个 batch

        max_len = max(len(feature['input_ids']) for feature in features)
        # 计算 当前 batch 内最长的 input_ids 长度，保证填充（padding）时对齐

        input_ids = []
        labels = []
        pixel_values = []
        # 创建列表，用于存储 填充后的 input_ids、labels 和 图像数据 pixel_values

        for feature in features:
            input_ids.append(
                feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))

            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            # 用 pad_token_id 进行填充，确保所有样本长度一致

            pixel_values.append(feature['pixel_values'])
            # 堆叠 pixel_values

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}



if __name__ == '__main__':  # 确保某些代码只在当前脚本被直接运行时执行，而在被导入时不会执行。
    config = AutoConfig.from_pretrained("vlm_model")  # 现在会返回 VLMConfig
    # config = VLMConfig(vision_model_path='/home/user/wyf/siglip-base-patch16-224', image_pad_num=49)

    model = AutoModel.from_config(config)  # 现在会返回 VLM 实例
    # model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False) # 从头开始训练
    trainer.save_model('save/pretrain') # 保存模型权重
    trainer.save_state() # 保存优化器状态等

