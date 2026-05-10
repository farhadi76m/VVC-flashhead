#!/usr/bin/env python3
"""
人脸裁剪处理脚本
从单张图像中检测人脸，裁剪并调整大小到指定尺寸
"""
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from flash_head.utils.cpu_face_handler import CPUFaceHandler

def get_scaled_bbox(
    bbox, img_w, img_h, ratio: float = 1.0, face_image: Image.Image = None
):
    """
    根据人脸边界框计算缩放后的裁剪区域
    
    Args:
        bbox: 人脸边界框 [x1, y1, x2, y2]
        img_w: 图像宽度
        img_h: 图像高度
        ratio: 缩放比例，数值越大，人脸在画面中的比例越小（周围留白越多）
        face_image: PIL Image 对象
    
    Returns:
        裁剪后的人脸图像
    """
    x1, y1, x2, y2 = bbox

    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate width and height
    width = x2 - x1

    # Scale width and height
    new_width = width * ratio
    new_height = new_width

    # tile pix
    dis_x_left = new_width * 0.5
    dis_x_right = new_width - dis_x_left  # 0.5new_width
    dis_y_up = new_height * 0.55
    dis_y_down = new_height - dis_y_up  # 0.45new_height

    # Calculate new coordinates
    new_x1 = int(max(0, center_x - dis_x_left))
    new_y1 = int(max(0, center_y - dis_y_up))
    new_x2 = int(min(img_w, center_x + dis_x_right))
    new_y2 = int(min(img_h, center_y + dis_y_down))
    scaled_bbox = [new_x1, new_y1, new_x2, new_y2]
    crop_face = face_image.crop(scaled_bbox)
    return crop_face


def process_image(
    input_path,
    face_ratio=2.0,
    target_size=(512, 512),
):
    """
    处理单张图像，进行人脸检测和裁剪
    
    Args:
        input_path: 输入图像路径
        face_ratio: 人脸缩放比例，建议范围：1.5-3.0，默认2.0
        target_size: 输出图像尺寸，默认(512, 512)
    
    Returns:
        imgae: 处理后的图像
    """
    # 初始化人脸检测器
    face_detector = CPUFaceHandler()
    
    # 验证输入文件
    if not os.path.isfile(input_path):
        raise ValueError(f"File not found: {input_path}")
    
    try:
        # 读取图像
        image = Image.open(input_path)
        image = image.convert("RGB")
        image_rgb = np.array(image)
        img_h, img_w = image_rgb.shape[:2]
        
        # 检测人脸
        boxes, scores = face_detector(image_rgb)
        
        if len(boxes) == 0:
            raise ValueError("No face detected")
        
        # 转换边界框坐标（从相对坐标转为绝对坐标）
        boxes_abs = [
            boxes[0][0] * img_w,
            boxes[0][1] * img_h,
            boxes[0][2] * img_w,
            boxes[0][3] * img_h
        ]
        
        # 裁剪人脸
        crop_face = get_scaled_bbox(boxes_abs, img_w, img_h, face_ratio, image)
        
        # 调整大小
        crop_face = crop_face.resize(target_size)
        
        return crop_face, image, boxes
            
    except Exception as e:
        raise ValueError(f"Error processing {input_path}: {e}")



def postprocess_image(video, original_image, boxes, face_ratio=2.0,):
    """
    Paste cropped/generated face back into original image using torch + CUDA.

    Args:
        video:
            Tensor (H, W, C) or (C, H, W)
            Example: (512, 512, 3)

        original_image:
            PIL image, numpy array, or tensor
            Original full frame.

        boxes:
            [x1, y1, x2, y2]

    Returns:
        torch.Tensor -> (H, W, C), uint8
    """

    
    original_image = torch.from_numpy(np.array(original_image)).to(video.device)
    original_image = original_image.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    
    x1, y1, x2, y2 = map(int, boxes)
    
    h = y2 - y1
    w = x2 - x1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 

    new_h = h * face_ratio
    new_w = w * face_ratio
    
    x1 = int(cx - new_w / 2)   
    y1 = int(cy - new_h / 2)
    x2 = int(cx + new_w / 2)
    y2 = int(cy + new_h / 2)

    target_h = y2 - y1
    target_w = x2 - x1

    video = video.permute(0,3,1,2) # (B, H, W, C) -> (B, C, H, W)
    video = F.interpolate(
        video,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    
    result = original_image.unsqueeze(0).repeat(video.shape[0], 1, 1, 1)
    result[:,:, y1:y2, x1:x2] = video
    result = result.clamp(0, 255).to(torch.uint8).permute(0,2,3,1)
    # from PIL import Image
    # Image.fromarray(result.cpu()[0].permute(1,2,0).numpy()).save('out.png')

    return result


def postprocess_image2(video, original_image, boxes, face_ratio=2.0):
    """
    boxes: raw relative coords [x1, y1, x2, y2] in range [0, 1]
           (as returned by face_detector, before the *img_w / *img_h conversion)
    """
    original_np = np.array(original_image)
    img_h, img_w = original_np.shape[:2]

    original_tensor = torch.from_numpy(original_np).to(video.device)
    original_tensor = original_tensor.permute(2, 0, 1)  # (C, H, W)

    # Convert relative → absolute, matching process_image exactly

    x1 = boxes[0][0] * img_w
    y1 = boxes[0][1] * img_h
    x2 = boxes[0][2] * img_w
    y2 = boxes[0][3] * img_h

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    # Mirror get_scaled_bbox exactly
    new_size    = w * face_ratio
    dis_x_left  = new_size * 0.5
    dis_x_right = new_size * 0.5
    dis_y_up    = new_size * 0.55
    dis_y_down  = new_size * 0.45

    px1 = int(max(0,     cx - dis_x_left))
    py1 = int(max(0,     cy - dis_y_up))
    px2 = int(min(img_w, cx + dis_x_right))
    py2 = int(min(img_h, cy + dis_y_down))

    target_h = py2 - py1
    target_w = px2 - px1

    video = video.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
    video = F.interpolate(video, size=(target_h, target_w),
                          mode="bilinear", align_corners=False)

    result = original_tensor.unsqueeze(0).repeat(video.shape[0], 1, 1, 1)
    result[:, :, py1:py2, px1:px2] = video
    result = result.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return result