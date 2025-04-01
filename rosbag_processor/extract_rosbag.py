import os
import argparse
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import PointCloud2
import re

# Topic types and names to process (configurable)
TOPICS = {
    '/sensors/camera/trunk_camera/aligned_depth_to_color/image_raw': 'sensor_msgs/msg/Image',
    '/sensors/camera/trunk_camera/color/image_raw': 'sensor_msgs/msg/Image',
    '/sensors/camera/trunk_camera/depth/color/points': 'sensor_msgs/msg/PointCloud2',
}

# Mapping of topic types to message classes
TYPES = {
    'sensor_msgs/msg/Image': ROS_Image,
    'sensor_msgs/msg/PointCloud2': PointCloud2,
}

class DataExtractor:
    def __init__(self, output_dir, interval=0.5):
        self.output_dir = output_dir
        self.bridge = CvBridge()
        self.interval = interval  # 时间窗口大小（秒）

        # Create folders to save extracted data
        self.rgb_folder = os.path.join(output_dir, 'rgb_images')
        self.depth_folder = os.path.join(output_dir, 'depth_images')
        self.pointcloud_folder = os.path.join(output_dir, 'pointclouds')

        for folder in [self.rgb_folder, self.depth_folder, self.pointcloud_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Dictionary to store the latest messages for each time window
        self.time_window_data = {}

    def extract_data(self, input_bag):
        # Open the rosbag2 file
        storage_options = rosbag2_py.StorageOptions(uri=input_bag, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Get the topics and their corresponding message types
        topic_type_dic = {topic_and_type.name: TYPES[topic_and_type.type]
                          for topic_and_type in reader.get_all_topics_and_types()}

        # Iterate through all messages in the rosbag2 file
        while reader.has_next():
            topic, data, _ = reader.read_next()

            if topic in TOPICS:
                msg_type = TOPICS[topic]
                timestamp_sec = deserialize_message(data, topic_type_dic[topic]).header.stamp.sec

                # Calculate the time window index
                time_window = timestamp_sec // self.interval

                # Initialize the dictionary for this time window
                if time_window not in self.time_window_data:
                    self.time_window_data[time_window] = {}

                # Store the latest message for this topic in the current time window
                if msg_type == 'sensor_msgs/msg/Image':
                    self.time_window_data[time_window][topic] = deserialize_message(data, topic_type_dic[topic])
                elif msg_type == 'sensor_msgs/msg/PointCloud2':
                    self.time_window_data[time_window][topic] = deserialize_message(data, topic_type_dic[topic])

        # Save aligned data after processing all messages
        self.save_aligned_data()

    def save_aligned_data(self):
        for time_window, data_dict in self.time_window_data.items():
            # Check if we have one message from each topic type
            rgb_msg = data_dict.get('/sensors/camera/trunk_camera/color/image_raw', None)
            depth_msg = data_dict.get('/sensors/camera/trunk_camera/aligned_depth_to_color/image_raw', None)
            pointcloud_msg = data_dict.get('/sensors/camera/trunk_camera/depth/color/points', None)

            if rgb_msg and depth_msg and pointcloud_msg:
                # Save aligned data using the timestamp of the RGB image
                timestamp = rgb_msg.header.stamp.sec
                print(f"Saving aligned data for time window {time_window} with timestamp {timestamp}")

                # Save RGB image
                self.save_image(rgb_msg, '/sensors/camera/trunk_camera/color/image_raw', timestamp)

                # Save depth image
                self.save_image(depth_msg, '/sensors/camera/trunk_camera/aligned_depth_to_color/image_raw', timestamp)

                # Save point cloud
                self.save_pointcloud(pointcloud_msg, '/sensors/camera/trunk_camera/depth/color/points', timestamp)

    def save_image(self, image_msg, topic, timestamp):
        try:
            # 正则表达式匹配
            if re.match(r'.*/color/image_raw$', topic):
                image_type = 'rgb'
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            elif re.match(r'.*/aligned_depth_to_color/image_raw$', topic):
                image_type = 'depth'
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
            else:
                print(f"Unknown image topic: {topic}")
                return

            # 使用时间戳命名文件
            filename = f"{timestamp}.png"

            # 确定保存路径
            if image_type == 'rgb':
                save_path = os.path.join(self.rgb_folder, filename)
            else:
                save_path = os.path.join(self.depth_folder, filename)

            # 保存图像
            cv2.imwrite(save_path, cv_image)
            print(f"Saved image: {save_path}")

        except Exception as e:
            print(f"Failed to process image from topic {topic}: {e}")

    def save_pointcloud(self, pointcloud_msg, topic, timestamp):
        try:
            # 使用时间戳命名文件
            filename = f"{timestamp}.pcd"
            save_path = os.path.join(self.pointcloud_folder, filename)

            # 转换 PointCloud2 为 PCD 格式（示例逻辑；你可能需要一个库如 open3d）
            with open(save_path, 'w') as f:
                f.write("# Placeholder for point cloud data\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Number of points: {len(pointcloud_msg.data)}\n")

            print(f"Saved point cloud: {save_path}")

        except Exception as e:
            print(f"Failed to process point cloud from topic {topic}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bag", "-i",
                        default="/home/yzyrobot/learning_data/factory_data/original_bag/rosbag2_2025_03_25-16_54_12",
                        type=str, help="Path to the input rosbag2 file", required=False)
    parser.add_argument("--output_dir", "-o",
                        default="/home/yzyrobot/learning_data/factory_data/extract/rosbag2_2025_03_25-16_54_12",
                        type=str, help="Output directory for extracted data")
    parser.add_argument("--interval", "-t", default=1, type=int,
                        help="Time interval (in seconds) for aligning data")
    args = parser.parse_args()

    # Initialize the data extractor
    data_extractor = DataExtractor(args.output_dir, interval=args.interval)

    # Extract data and save
    data_extractor.extract_data(args.input_bag)


if __name__ == "__main__":
    main()