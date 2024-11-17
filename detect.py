# 还有就是用模型识别图片视频啥的没什么精度所以识别不到
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\ultralytics-main\ultralytics\runs\train\exp6\weights\best.pt'
                 r''
                 r''
                 r'')  # YOLOv8n模型 加载自己模型
    model.predict(
        source='E:\images', # 待识别的文件具体位置
        # source='E:/datasets/VisDrone/images/test',
        # source='E:/datasets/VisDrone/images/test',
        # source='E:/datasets/VisDrone/images/test',
        save=True,  # 保存预测结果
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project='runs/detect',  # 项目名称（可选）
        name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
        save_txt=False,  # 保存结果为 .txt 文件
        save_conf=True,  # 保存结果和置信度分数
        save_crop=False,  # 保存裁剪后的图像和结果
        show_labels=True,  # 在图中显示目标标签
        show_conf=True,  # 在图中显示目标置信度分数
        vid_stride=1,  # 视频帧率步长
        line_width=3,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        max_det=1000,  # 每张图像的最大检测数
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=False,  # 使用高分辨率的分割掩码
        boxes=True,  # 在分割预测中显示边界框
    )

