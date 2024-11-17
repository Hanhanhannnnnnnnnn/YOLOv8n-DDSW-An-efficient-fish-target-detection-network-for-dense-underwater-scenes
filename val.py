
# 训练好了模型就要验证性能 就用这个
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\ultralytics-main\ultralytics\runs\train\exp3\weights\best.pt') #这里路径是要验证的模型，自己训练后最好权重的路径
    # 验证模型
    model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data=r'D:/ultralytics-main/data/my.yaml',  #数据集配置文件所在路径
        split='val',  # (str) 设置为什么就是看什么集的结果，用于验证的数据集拆分，例如'val'、'test'或'train'  #三个 一般就val 对应yaml文件里的
        batch=1,  # 测试时设置为1比较严谨，(int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device='0',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=16,  # 数据加载的工作线程数（每个DDP进程）
        save_json=False,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='val/AI-TOD',  # 项目名称（可选）
        name='ours',  # 实验名称，结果保存在'project/name'目录下（可选） 这里也是路径 然后就可以了 当然肯定是没什么精度的现在 有点慢 下面就是精度了 P R MAP 50这些字面精度
        max_det=1000,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
    )
