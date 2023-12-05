from PIL import Image
import os
from tqdm import tqdm
import random
import shutil

def crop_images_in_folder():
    # 输入大图片文件夹路径和输出文件夹路径
    input_folder = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise\GT'
    output_folder = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise\GT_crop'
    # 目标裁剪尺寸
    crop_size = (512, 512)
    # 遍历文件夹中的每张图片
    for count, filename in tqdm(enumerate(os.listdir(input_folder)), desc="Processing Images"):
    # for count, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".png"):
            # 读取原始图片
            input_path = os.path.join(input_folder, filename)
            filename, file_extension = os.path.splitext(filename)
            image = Image.open(input_path)

            # 获取图片大小
            width, height = image.size

            # 定义滑动窗口的步幅
            stride = 256

            # 遍历图像并裁剪
            cnt = 1
            for y in range(0, height - crop_size[1] + 1, stride):
                for x in range(0, width - crop_size[0] + 1, stride):
                    # 计算窗口的坐标
                    left, top, right, bottom = x, y, x + crop_size[0], y + crop_size[1]

                    # 裁剪图像
                    cropped_image = image.crop((left, top, right, bottom))

                    # 生成新的文件名，格式为filename_数字编号.png
                    new_filename = f"{filename}_{cnt}.png"
                    cnt += 1
                    output_path = os.path.join(output_folder, new_filename)
                    cropped_image.save(output_path)

    print("裁剪和重命名完成。")

def rename_files_in_folder():
    # 输入文件夹路径和输出文件夹路径
    input_folder = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise\Image'
    output_folder = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise\Image'
    # 遍历文件夹中的每个文件
    for count, filename in enumerate(os.listdir(input_folder)):
        # 生成新的文件名，格式为image_数字编号.png
        new_filename = f"{count}.png"
        
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, new_filename)
        
        # 重命名文件
        os.rename(input_path, output_path)

    print("重命名完成。")

def split_dataset():
    # 设置数据集目录和划分比例
    dataset_dir = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise\Image'
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    random.seed(42)  # 设置随机种子

    # 获取数据集中所有图片的文件名列表
    all_images = os.listdir(dataset_dir)
    random.shuffle(all_images)

    # 计算划分的索引
    total_images = len(all_images)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))

    # 划分数据集
    train_set = all_images[:train_split]
    val_set = all_images[train_split:val_split+1]
    test_set = all_images[val_split+1:]

    # 定义保存划分结果的目录
    split_dir = 'E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise'
    os.makedirs(split_dir, exist_ok=True)

    # 将图片复制到相应的目录
    def copy_images(image_list, destination_dir):
        for image in image_list:
            source_path = os.path.join(dataset_dir, image)
            destination_path = os.path.join(destination_dir, image)
            shutil.copyfile(source_path, destination_path)

    copy_images(train_set, os.path.join(split_dir, 'train\GT'))
    copy_images(val_set, os.path.join(split_dir, 'val\GT'))
    copy_images(test_set, os.path.join(split_dir, 'test\GT'))


if __name__ == '__main__':
    # split_dataset()
    '''
    list1 = os.listdir('E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise/test\Image')
    list2 = os.listdir('E:\Deep Learning\IATlab\Image-Reconstruction\datasets\SenseNoise/test\GT')

    # # 使用集合交集操作符 "&" 找到两个列表的交集
    intersection = list(set(list1) & set(list2))

    print("两个列表的交集是:", len(intersection))
    '''