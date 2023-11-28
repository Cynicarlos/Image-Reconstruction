import torch


def block_process(image, model,block_size=(576,576), overlap=(24, 24)):
    with torch.no_grad():
        _, _, height, width = image.shape
        block_height, block_width = block_size
        stride_y, stride_x = block_size
        overlap_y, overlap_x = overlap
        # 初始化输出图像
        processed_image = torch.zeros_like(image)

        for y in range(0, height - block_height + 1, stride_y - overlap_y):
            for x in range(0, width - block_width + 1, stride_x - overlap_x):
                block = image[:, :, y:y + block_height, x:x + block_width]
                processed_block = model(block)
                processed_image[:, :, y:y + block_height, x:x + block_width] += processed_block
    return processed_image
