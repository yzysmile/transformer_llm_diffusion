import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms

from viplanner.config.learning_cfg import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder, get_m2f_cfg
from viplanner.traj_cost_opt.traj_opt import TrajOpt

torch.set_default_dtype(torch.float32)

# 自添加模块
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Quaternion
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer

from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration


import rclpy
from geometry_msgs.msg import PoseStamped, Point, Quaternion


from cv_bridge import CvBridge
import cv2
import struct


class VIPlannerInference:
    def __init__(
        self,
        cfg,
    ) -> None:
        """VIPlanner Inference Script

        Args:
            cfg (Namespace): Config Namespace
        """
        # get configs
        model_path = os.path.join(cfg.model_save, "model.pt")
        config_path = os.path.join(cfg.model_save, "model.yaml")

        # get train config
        self.train_cfg: TrainCfg = TrainCfg.from_yaml(config_path)

        # get model
        if self.train_cfg.rgb:
            m2f_cfg = get_m2f_cfg(cfg.m2f_cfg_path)
            self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
            self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
        else:
            m2f_cfg = None
            self.pixel_mean = [0, 0, 0]
            self.pixel_std = [1, 1, 1]

        if self.train_cfg.rgb or self.train_cfg.sem:
            self.net = DualAutoEncoder(train_cfg=self.train_cfg, m2f_cfg=m2f_cfg)  # self.net只是网络结构
        else:
            self.net = AutoEncoder(
                encoder_channel=self.train_cfg.in_channel,
                k=self.train_cfg.knodes,
            )
        try:
            model_state_dict, _ = torch.load(model_path)
        except ValueError:
            model_state_dict = torch.load(model_path)
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self._device = "cuda"
        else:
            self._device = "cpu"

        # transforms
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(tuple(self.train_cfg.img_input_size)),
            ]
        )

        # get trajectory generator
        self.traj_generate = TrajOpt()
        return

    def img_converter(self, img: np.ndarray) -> torch.Tensor:
        # crop image and convert to tensor
        img = self.transforms(img)
        return img.unsqueeze(0).to(self._device)

    def plan(
        self,
        depth_image: np.ndarray,
        sem_rgb_image: np.ndarray,
        goal_robot_frame: torch.Tensor,  # 目标在机器人坐标系的表示
    ) -> tuple:
        """Plan to path towards the goal given depth and semantic image

        Args:
            depth_image (np.ndarray): Depth image from the robot
            goal_robot_frame (torch.Tensor): Goal in robot frame
            sem_rgb_image (np.ndarray): Semantic/ RGB Image from the robot.

        Returns:
            tuple: _description_
        """

        with torch.no_grad():
            depth_image = self.img_converter(depth_image).float()
            if self.train_cfg.rgb:
                sem_rgb_image = (sem_rgb_image - self.pixel_mean) / self.pixel_std
            sem_rgb_image = self.img_converter(sem_rgb_image.astype(np.uint8)).float()
            keypoints, fear = self.net(depth_image, sem_rgb_image, goal_robot_frame.to(self._device))

        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()

    def plan_depth(
        self,
        depth_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        with torch.no_grad():
            depth_image = self.img_converter(depth_image).float()
            keypoints, fear = self.net(depth_image, goal_robot_frame.to(self._device))

        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()

class Viplanner(Node):
    def __init__(self, args, semantic_qos ,depth_qos):
        super().__init__('vip')

        # 订阅语义分割图像 和 深度图像
        self.depth_sub = Subscriber(self, Image, '/middle_camera/aligned_depth_to_color/image_raw', qos_profile = depth_qos)
        self.semantic_sub = Subscriber(self, Image, '/middle_camera/color/image_semantic', qos_profile = semantic_qos)

        # self.sync = TimeSynchronizer([self.depth_sub, self.semantic_sub], 10)

        self.time_dif_threshold = 0.1
        self.sync = ApproximateTimeSynchronizer([self.depth_sub, self.semantic_sub], queue_size=10, slop=self.time_dif_threshold)
        self.sync.registerCallback(self.sync_callback)

        self.bridge = CvBridge()

        self.goal_cam_frame = torch.tensor([1.5, 0.0, 0.0])

        self.vip_algo = VIPlannerInference(args)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_msg = Path()
        self.path_pub = self.create_publisher(Path, '/middle_camera/path',10)

    def transform_waypoints(self):
        transformed_waypoints = []
        try:
            # 获取从 middle_camera_link 到 base_link 的变换
            trans = self.tf_buffer.lookup_transform('base_link', 'middle_camera_link', rclpy.time.Time())
            for waypoint in self.waypoints:
                point = Point()

                point.x = float(waypoint[0])
                point.y = float(waypoint[1])
                point.z = float(waypoint[2])
                # 创建一个 PointStamped 对象
                point_stamped = PointStamped()
                point_stamped.point = point
                point_stamped.header.frame_id = 'middle_camera_link'  # 当前点所在的参考系

                # 应用变换
                transformed_point_stamped = do_transform_point(point_stamped, trans)

                # 将变换后的 PointStamped 转回 Point
                transformed_point = transformed_point_stamped.point
                transformed_waypoints.append(transformed_point)
        except Exception as e:
            self.get_logger().error(f"变换错误: {e}")
        return transformed_waypoints

    def create_path_message(self, transformed_waypoints):
        path_msg = Path()
        path_msg.header.frame_id = "base_link"  # 设置帧 ID 为 base_link
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for waypoint in transformed_waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = waypoint
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # 默认方向
            path_msg.poses.append(pose)

        return path_msg

    def publish_waypoints(self):
        transformed_waypoints = self.transform_waypoints()
        if transformed_waypoints:
            path_msg = self.create_path_message(transformed_waypoints)
            self.path_pub.publish(path_msg)

    def sync_callback(self, depth_msg, semantic_msg):
        depth_time = depth_msg.header.stamp
        semantic_time = semantic_msg.header.stamp

        time_diff = abs((depth_time.sec + depth_time.nanosec * 1e-9) - (semantic_time.sec + semantic_time.nanosec * 1e-9))
        # self.get_logger().info(f"Time difference: {time_diff} seconds.")

        if time_diff < self.time_dif_threshold:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth = depth.astype(np.float32)/1000.0 # mm->m
            semantic = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='bgr8')

            goal_cam_frame_extension = self.goal_cam_frame[None, :]
            self.waypoints, fear = self.vip_algo.plan(depth, semantic, goal_cam_frame_extension)

            # 将self.waypoints(ndarray)以 from nav_msgs.msg import Path的格式 发布到base_link坐标系
            self.publish_waypoints()


def main(args=None):
    parser = argparse.ArgumentParser()
    # networks(与加载模型相关的参数)
    parser.add_argument(
        "--model_save",  # 模型保存目录，包含模型文件model.pt和配置文件model.yaml
        type=str,
        default="/SSD/yzy/mi_car/model",
        help=("model directory (within should be a file called model.pt and" " model.yaml)"),
    )

    # 参数添加
    parser.add_argument(
        "--m2f_cfg_path",  # 用于Mask2Former模型（或直接RGB输入的预训练骨干网络）的配置文件路径
        type=str,
        default=("/SSD/yzy/mi_car/ros/planner/src/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"),
        help=("config file for m2f model (or pre-trained backbone for direct RGB" " input)"),
    )

    # 参数解析
    args = parser.parse_args()
    rclpy.init()

    semantic_qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,  # 或 QoSHistoryPolicy.KEEP_ALL
        depth=10,  # 消息队列的大小
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE
    )

    depth_qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,  # 或 QoSHistoryPolicy.KEEP_ALL
        depth=10,  # 消息队列的大小
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE
    )

    vip = Viplanner(args, semantic_qos, depth_qos)
    print("Loaded vip")
    rclpy.spin(vip)
    vip.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

