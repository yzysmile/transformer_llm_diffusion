import numpy as np
from PIL import Image, ImageDraw
import json
import open3d as o3d
import argparse
import os
from tqdm import tqdm
import cv2

#############################################################################################
# Generate 3D point cloud data from 2D images and annotation files, and process and save it #
#############################################################################################

def parse_args():

    parser = argparse.ArgumentParser(description="Data annotation")
    parser.add_argument('--file-path', default='/home/yzyrobot/learning_data/factory_data/extract/rosbag2_2025_03_25-16_54_12/', type=str,
                        help='Label path')
    parser.add_argument('--down-sample', default=True, type=bool,
                        help='Enable image downsampling')
    parser.add_argument('--idx', default=0, type=int,
                        help='Image downsampling index, default=0, (range: 0-7)')
    return parser.parse_args()

def get_json(filepath):
    """
    Loads and parses JSON data from a file.
    
    Parameters:
    filepath (str): Path to the JSON file.

    Returns:
    dict: Parsed JSON data.
    """
    with open(filepath, "r", encoding="utf8") as file:
        return json.load(file)

def separate_ground_point(points_list, width, height, down_sample=False):
    """
    Segments ground and non-ground points based on annotated points list.

    Parameters:
    points_list (list): List of 2D points representing annotations.
    width (int): Image width.
    height (int): Image height.
    down_sample (bool): Flag for downsampling.

    Returns:
    tuple: Arrays of ground and non-ground points.
    """
    blank_image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(blank_image)
    for points in points_list:
        xy = [tuple(point) for point in points]
        draw.polygon(xy, outline="white", fill="white")
    image_array = np.array(blank_image)

    # 可视化image_array 验证白色区域是否为地面点云
    # cv2.imshow("Image Array", image_array.astype('uint8'))
    # cv2.waitKey(0)  # 等待用户按键
    # cv2.destroyAllWindows()  # 关闭窗口

    if down_sample:
        image_array = pixel_interval_downsample(image_array, 8)

    ground_points = np.column_stack(np.where(image_array == 255))  # 白色区域为地面点云
    nonground_points = np.column_stack(np.where(image_array != 255))
    return ground_points, nonground_points

def read_camera_parameters():
    """
    Returns predefined camera parameters for RGB and depth images.

    Returns:
    dict: Camera intrinsic and extrinsic parameters.
    """
    return {
        "rgb_to_depth_rotation": [-0.00085284, -0.00132613, -0.00128921, 0.99999793],
        "rgb_to_depth_translation": [-0.0591784, -0.00014266, -0.00081757],
        "depth_intrinsics": [[427.36554381, 0, 421.67051326], [0, 427.59220613, 239.69789247], [0, 0, 1]],
        "rgb_intrinsics": [[636.60212122, 0, 635.19429872], [0, 636.43281793, 372.82353189], [0, 0, 1]]
    }

def depth_and_rgb_to_point_cloud(pixels, depth_image, rgb_image, fx, fy, cx, cy, ground=True, idx=0):
    """
    Converts depth and RGB pixel data to 3D point cloud.

    Parameters:
    pixels (array): Array of pixels to process.
    depth_image (ndarray): Depth image.
    rgb_image (ndarray): RGB image.
    fx, fy, cx, cy (float): Intrinsic camera parameters.
    ground (bool): Flag for ground labeling.
    idx (int): Downsampling index.

    Returns:
    ndarray: Point cloud with color(rgb) and label data.
    """
    point_cloud = []
    label = 0 if ground else 1
    for per_pixel in pixels:
        u, v = per_pixel[1]*8 + idx, per_pixel[0]*8 + idx
        z = depth_image[v, u] / 1000
        x, y = (u - cx) * z / fx, (v - cy) * z / fy
        r, g, b = rgb_image[v, u]
        point_cloud.append([x, y, z, r, g, b, label])
    return np.array(point_cloud)

def noise_filter(points):
    """
    Filters noise in the point cloud using RANSAC for plane segmentation.

    Parameters:
    points (ndarray): Input point cloud data.

    Returns:
    list: Indices of points considered as inliers.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    _, inliers = point_cloud.segment_plane(distance_threshold=0.08, ransac_n=3, num_iterations=1000)
    return inliers

def show_point_cloud(points, inliers_index=None):
    """
    Visualizes the point cloud with optional inliers highlighting.

    Parameters:
    points (ndarray): Point cloud data.
    inliers_index (list, optional): Indices for inliers.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    if inliers_index:
        inlier_cloud = point_cloud.select_by_index(inliers_index)
        inlier_cloud.paint_uniform_color([1, 0, 0])
        outlier_cloud = point_cloud.select_by_index(inliers_index, invert=True)
        outlier_cloud.paint_uniform_color([0, 0, 1])

def pixel_interval_downsample(image, factor):
    """
    Downsamples an image by a given factor.

    Parameters:
    image (ndarray): Input image.
    factor (int): Downsampling factor.

    Returns:
    ndarray: Downsampled image.
    """
    return image[::factor, ::factor]

def load_data(args):
    """
    Loads data paths and camera parameters.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.

    Returns:
    tuple: Contains file paths, image names, and camera parameters fx, fy, cx, cy.
    """
    file_path = args.file_path
    rgb_path = os.path.join(file_path, "rgb_images/")
    depth_path = os.path.join(file_path, "depth_images/")
    json_path = os.path.join(file_path, "rgb_images_label/")
    image_names = os.listdir(rgb_path)
    
    camera_info = read_camera_parameters()
    fx = camera_info["rgb_intrinsics"][0][0]
    fy = camera_info["rgb_intrinsics"][1][1]
    cx = camera_info["rgb_intrinsics"][0][2]
    cy = camera_info["rgb_intrinsics"][1][2]
    
    return file_path, rgb_path, depth_path, json_path, image_names, fx, fy, cx, cy

def delete_png_images(folder_path):
    """
    删除指定路径下的所有 .png 图像文件。

    参数:
        folder_path (str): 文件夹路径。
    """
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    # 遍历文件夹中的所有文件
    deleted_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否是文件且以 .png 结尾
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除失败: {file_path} ({e})")

def label_rgb_depth_alignment(rgb_path, depth_path, json_path):
    """
    根据 json_path 中的 .json 文件名称，仅保留 rgb_path 和 depth_path 中相同名称的 .png 文件。

    参数:
        rgb_path (str): RGB 图像文件夹路径。
        depth_path (str): 深度图像文件夹路径。
        json_path (str): JSON 文件夹路径。
    """
    # 检查路径是否存在
    if not os.path.exists(rgb_path):
        print(f"RGB 路径不存在: {rgb_path}")
        return
    if not os.path.exists(depth_path):
        print(f"深度图像路径不存在: {depth_path}")
        return
    if not os.path.exists(json_path):
        print(f"JSON 路径不存在: {json_path}")
        return

    # 获取 json_path 中所有 .json 文件的基础名称（不带扩展名）
    json_filenames = {
        os.path.splitext(filename)[0]  # 去掉扩展名
        for filename in os.listdir(json_path)
        if filename.lower().endswith('.json')
    }

    # 遍历 rgb_path 和 depth_path，删除不符合条件的文件
    def align_folder(folder_path, file_extension):
        deleted_count = 0
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(file_extension):
                continue  # 忽略非目标扩展名的文件

            base_name = os.path.splitext(filename)[0]  # 去掉扩展名
            file_path = os.path.join(folder_path, filename)

            # 如果文件名不在 json_filenames 中，则删除
            if base_name not in json_filenames:
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败: {file_path} ({e})")

        print(f"在 {folder_path} 中共删除了 {deleted_count} 个文件。")

    # 对齐 RGB 和 Depth 文件夹
    align_folder(rgb_path, '.png')
    align_folder(depth_path, '.png')

def main():
    """
    Main function to process images, generate point clouds, and save results.
    """
    args = parse_args()
    file_path, rgb_path, depth_path, json_path, image_names, fx, fy, cx, cy = load_data(args)
    # os.makedirs(file_path + "3d_annotation", exist_ok=True)

    # 删除json_path文件夹下的图像，仅保留.json
    delete_png_images(json_path)

    # rgb_path、depth_path中的.png 与 json_path中的.json对齐
    label_rgb_depth_alignment(rgb_path, depth_path, json_path)

    for image_name in tqdm(image_names, total=len(image_names)):
        rgb_image = np.array(Image.open(os.path.join(rgb_path, image_name)))
        depth_image = np.array(Image.open(os.path.join(depth_path, image_name)))
        points_data = get_json(os.path.join(json_path, image_name.split('.')[0] + ".json"))

        points_list = [shape["points"] for shape in points_data["shapes"]]
        width, height = depth_image.shape[1], depth_image.shape[0]

        # 根据.json划分 地面 或 障碍物
        ground_pixels, nonground_pixels = separate_ground_point(points_list, width, height, args.down_sample)

        ground_points = depth_and_rgb_to_point_cloud(ground_pixels, depth_image, rgb_image, fx, fy, cx, cy, True, args.idx)
        nonground_points = depth_and_rgb_to_point_cloud(nonground_pixels, depth_image, rgb_image, fx, fy, cx, cy, False, args.idx)
        
        whole_points = np.concatenate((ground_points, nonground_points))
        z_index = np.where(whole_points[:, 2] < 7)[0]
        whole_points = whole_points[z_index]

        # 可视化点云
        pred_ground_pcd = o3d.geometry.PointCloud()
        pred_nonground_pcd = o3d.geometry.PointCloud()

        # 地面点云
        pred_ground_pcd.points = o3d.utility.Vector3dVector(ground_points[:, :3])
        pred_ground_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

        # 非地面点云
        pred_nonground_pcd.points = o3d.utility.Vector3dVector(nonground_points[:, :3])
        pred_nonground_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色

        # 合并地面和非地面点云
        combined_pcd = pred_ground_pcd + pred_nonground_pcd

        # 保存为 .ply 文件，以可视化 标注的是否正确
        save_dir = os.path.join(args.file_path, 'sample_points/')
        os.makedirs(save_dir, exist_ok=True)

        ply_file_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_{args.idx}.ply")
        o3d.io.write_point_cloud(ply_file_path, combined_pcd)
        print(f"Saved point cloud to {ply_file_path}")

        # 保存为 .npz 文件，训练的原始数据
        np.savez_compressed(os.path.join(save_dir, f"{image_name.split('.')[0]}_{args.idx}.npz"),
                            point_clouds=whole_points)
if __name__ == "__main__":
    main()
    # 查看文件中包含的所有数组名称
    # print(data.files)


