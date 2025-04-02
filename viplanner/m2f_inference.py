# python
import numpy as np
import torch

# ROS
# import rospy
from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation import INSTANCE_OFFSET

# viplanner-ros
from viplanner.config.coco_sem_meta import get_class_for_id_mmdet
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False

    def __init__(
        self,
        config_file="Base-COCO-PanopticSegmentation",
        checkpoint_file="model_final.pth",
    ) -> None:
        # Build the model from a config file and a checkpoint file
        # config文件定义 模型结构，checkpoint_file文件 保存模型权重
        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")

        # mapping from coco class id to viplanner class id and color
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]

        return

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict semantic segmentation from image

        Args:
            image (np.ndarray): image to be processed in BGR format
        """

        result = inference_detector(self.model, image)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        # create output
        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            curr_label = curr_sem_class % INSTANCE_OFFSET
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_label]
            except KeyError:
                if curr_sem_class != len(self.model.dataset_meta["classes"]):
                    print(f"Category {curr_label} not found in" " coco_viplanner_cls_mapping.")
                    # rospy.logwarn(f"Category {curr_label} not found in" " coco_viplanner_cls_mapping.")
                panoptic_mask[result == curr_sem_class] = self.viplanner_sem_class_color_map["static"]

        if self.debug:
            import matplotlib.pyplot as plt

            plt.imshow(panoptic_mask)
            plt.show()

        return panoptic_mask
# EoF

class Mask2former(Node):
    def __init__(self, qos):
        super().__init__('m2f')

        self.m2f = Mask2FormerInference(config_file="/SSD/yzy/mi_car/ros/planner/src/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",                                                       
                          checkpoint_file="/SSD/yzy/mi_car/ros/planner/src/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth")

        self.subscription = self.create_subscription(
            Image,
            '/middle_camera/color/image_raw',
            self.rgb_semantic_callback,
            qos
        )
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/middle_camera/color/image_semantic', 10)

    def rgb_semantic_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ndarray_image = np.array(cv_image)
        ndarray_semantic = self.m2f.predict(ndarray_image)
        ros_semantic = self.bridge.cv2_to_imgmsg(ndarray_semantic, encoding='bgr8')
        # 时间戳赋值
        ros_semantic.header.stamp = self.get_clock().now().to_msg()

        # 发布图像消息
        self.publisher.publish(ros_semantic)

def main(args=None):
    rclpy.init()

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,  # 或 QoSHistoryPolicy.KEEP_ALL
        depth=10,  # 消息队列的大小
        reliability=QoSReliabilityPolicy.BEST_EFFORT,  # QoSReliabilityPolicy.RELIABLE
        durability=QoSDurabilityPolicy.VOLATILE
    )

    m2f = Mask2former(qos)
    print("Loaded m2f")
    rclpy.spin(m2f)
    m2f.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

