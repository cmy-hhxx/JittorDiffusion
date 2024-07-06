import os
from PIL import Image
import math
import random
from tqdm import tqdm

def random_rotation(image):
    """对图像进行轻微的旋转，并适当放大以去除边角的黑色填充。"""
    # 设置旋转角度范围
    angle = random.uniform(-8, 8)
    # 旋转图像，使用expand=True让图像尺寸适应新的界限
    rotated_image = image.rotate(angle, expand=True)

    # 计算放大的比例，确保没有黑色填充
    # 因为旋转最大可能在角落形成空白，放大的比例由余弦定理决定
    angle_rad = math.radians(abs(angle))
    if angle_rad != 0:
        # 根据余弦定理计算对角线和宽高比，确定最小的放大比例
        cos_angle = math.cos(angle_rad)
        scale_factor = 1 / cos_angle
    else:
        scale_factor = 1

    # 放大图像
    new_width = int(rotated_image.width * scale_factor)
    new_height = int(rotated_image.height * scale_factor)
    scaled_image = rotated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 裁剪图像到原始尺寸
    center_x, center_y = scaled_image.width // 2, scaled_image.height // 2
    original_width, original_height = image.size
    box = (
        center_x - original_width // 2,
        center_y - original_height // 2,
        center_x + original_width // 2,
        center_y + original_height // 2
    )
    cropped_image = scaled_image.crop(box)

    return cropped_image

def random_flip(image):
    """随机进行水平或垂直翻转。"""
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def random_zoom(image):
    """对图像随机放大部分区域，并缩放回原始尺寸。"""
    width, height = image.size
    zoom_scale = random.uniform(1.1, 2.5)  # 放大比例
    x1 = random.randint(0, int(width * (1 - 1/zoom_scale)))
    y1 = random.randint(0, int(height * (1 - 1/zoom_scale)))
    cropped_image = image.crop((x1, y1, x1 + int(width/zoom_scale), y1 + int(height/zoom_scale)))
    return cropped_image.resize(image.size, Image.LANCZOS)

def augment_image(image_path, save_dir, num_augmented=9):
    """
    对单一图片进行数据增强操作，包括旋转、裁剪，并缩放回原始尺寸。
    :param image_path: 原始图片路径。
    :param save_dir: 增强后图片保存的目录。
    :param num_augmented: 生成的增强图片数量。
    """
    try:
        # 加载原始图片
        img = Image.open(image_path)

        # 获取原始文件名，不包括扩展名，并保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        img.save(os.path.join(save_dir, f"{base_name}_original.png"))

        for i in range(1, num_augmented + 1):
            augmented_img = img.copy()
            augmented_img = random_rotation(augmented_img)
            augmented_img = random_flip(augmented_img)
            augmented_img = random_zoom(augmented_img)
            augmented_img.save(os.path.join(save_dir, f"{base_name}_{i}.png"))

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def augment_images_in_directory(directory, num_augmented=9):
    """
    对指定目录中的所有图片进行数据增强操作。
    :param directory: 包含图片的目录。
    :param num_augmented: 每张图片生成的增强图片数量。
    """
    try:
        for filename in tqdm(os.listdir(directory), desc="Processing images"):
            if filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                save_dir = os.path.join(directory, "augmented")
                os.makedirs(save_dir, exist_ok=True)  # 创建保存增强图片的目录

                augment_image(image_path, save_dir, num_augmented)
    except Exception as e:
        print(f"Error augmenting images in {directory}: {e}")


def main():
    base_dir = "./A"
    num_styles = 15
    for style_id in tqdm(range(num_styles), desc="Processing styles"):
        style_dir = os.path.join(base_dir, f"{style_id:02}", "images")
        augment_images_in_directory(style_dir)


if __name__ == "__main__":
    main()
