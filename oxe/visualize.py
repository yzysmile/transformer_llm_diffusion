import tensorflow as tf
import tensorflow_datasets as tfds
import os
import cv2
from PIL import Image
import json

# 使用 builder_from_directory 从现有目录构建数据集
builder_dir = "/home/yzyrobot/Downloads/oxe/data/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0"
builder = tfds.builder_from_directory(builder_dir=builder_dir)

# 加载训练集
ds = builder.as_dataset(split='train')
stop = 1

# 遍历数据集并处理每个 episode
for idx, episode in enumerate(ds):  # 使用 enumerate 获取索引
    print(f"处理第 {idx + 1} 条 episode")

    # 获取文件路径，作为目录名的一部分
    episode_file_path = episode['episode_metadata']['file_path'].numpy().decode()

    # 创建基于文件路径的文件夹（去掉特殊字符）
    folder_name = episode_file_path.replace('/', '_').replace(' ', '_')  # 文件夹名不能包含 / 或空格
    save_folder_path = os.path.join(builder_dir, folder_name)

    # 确保保存路径存在
    os.makedirs(save_folder_path, exist_ok=True)

    steps = episode['steps']  # 获取 episode 中的 steps

    # 初始化一个视频写入器（cv2.VideoWriter）
    video_filename = f"episode_{idx + 1}.mp4"  # 视频文件名
    video_save_path = os.path.join(save_folder_path, video_filename)

    # 假设所有图像的尺寸都相同，获取第一个图像的尺寸
    first_step = next(iter(steps))  # 获取第一个 step
    first_image = first_step['observation']['image'].numpy()
    height, width, _ = first_image.shape

    # 定义视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    video_writer = cv2.VideoWriter(video_save_path, fourcc, 30.0, (width, height))  # 30 FPS

    step_data_list = []  # 用于保存每一步的详细数据（可选：存 json）

    # 遍历每个 step，将图像添加到视频
    for step_idx, step in enumerate(steps):
        data = {
            "step_index": step_idx,
            "action": step["action"].numpy().tolist(),
            "discount": float(step["discount"].numpy()),
            "is_first": bool(step["is_first"].numpy()),
            "is_last": bool(step["is_last"].numpy()),
            "is_terminal": bool(step["is_terminal"].numpy()),
            "language_instruction": step["language_instruction"].numpy().decode(),
            "language_embedding": step["language_embedding"].numpy().tolist(),
            "reward": float(step["reward"].numpy()),
            "observation": {
                "state": step["observation"]["state"].numpy().tolist(),
            }
        }

        # 图像单独处理（保存图像或写入视频）
        image = step["observation"]["image"].numpy()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_bgr)

        # 可选：保存单帧图像
        img_path = os.path.join(save_folder_path, f"frame_{step_idx:03d}.png")
        cv2.imwrite(img_path, image_bgr)

        # 添加到数据列表
        step_data_list.append(data)

    video_writer.release()
    print(f"视频已保存到: {video_save_path}")

    # 保存所有 step 的信息为 json（可选）
    json_path = os.path.join(save_folder_path, f"episode_{idx + 1}_steps.json")
    with open(json_path, "w") as f:
        json.dump(step_data_list, f, indent=2)

    print(f"数据已保存到: {json_path}")
