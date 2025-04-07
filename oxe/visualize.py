import tensorflow as tf
import tensorflow_datasets as tfds
import os
import cv2
from PIL import Image

# 使用 builder_from_directory 从现有目录构建数据集
builder_dir = "/home/yzyrobot/Downloads/oxe/cmu_stretch/0.1.0"
builder = tfds.builder_from_directory(builder_dir=builder_dir)

# 加载训练集
ds = builder.as_dataset(split='train')

# FeaturesDict({
#     'episode_metadata': FeaturesDict({
#         'file_path': Text(shape=(), dtype=string),
#     }),
#     'steps': Dataset({
#         'action': Tensor(shape=(8,), dtype=float32, description=Robot action, consists of [3x end effector pos, 3x robot rpy angles, 1x gripper open/close command, 1x terminal action].),
#         'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
#         'is_first': bool,
#         'is_last': bool,
#         'is_terminal': bool,
#         'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
#         'language_instruction': Text(shape=(), dtype=string),
#         'observation': FeaturesDict({
#             'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
#             'state': Tensor(shape=(7,), dtype=float32, description=Robot state, consists of [3x end effector pos, 3x robot rpy angles, 1x gripper position].),
#         }),
#         'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
#     }),
# })

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

    # 遍历每个 step，将图像添加到视频
    for step in steps:
        action = step['action'].numpy()
        image = step['observation']['image'].numpy()
        state = step['observation']['state'].numpy()

        # step是一个字典，
        # key包括"action"、"discount"、"is_first"、'is_last'、'is_terminal'、 'language_embedding'、'language_instruction'、'observation'、'reward'
        # 'observation'中还有两个key，分别是 'image'、'state'(机器人本体状态， robot joint angles, joint velocity and joint torque)
        #                                                               [3x end effector pos, 3x robot rpy angles, 1x gripper position]

        # 打印图像形状
        # print(f"图像形状: {image.shape}")

        # 将图像转换为 RGB 格式（OpenCV 使用 BGR）
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 写入视频
        video_writer.write(image_bgr)

    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存到: {video_save_path}")
