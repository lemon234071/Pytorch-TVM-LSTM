import torch
import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image

from pytorch-mobilenet-v2.MobileNetv2 import mobilenetv2

model = mobilenetv2(pretrained=True)
example = torch.rand(1, 3, 224, 224)  # 假想输入

torch_out = torch.onnx.export(model,
                              example,
                              "mobilenetv2.onnx",
                              verbose=True,
                              export_params=True  # 带参数输出
                              )

onnx_model = onnx.load('mobilenetv2.onnx')  # 导入模型

mean = [123., 117., 104.]  # 在ImageNet上训练数据集的mean和std
std = [58.395, 57.12, 57.375]


def transform_image(image):  # 定义转化函数，将PIL格式的图像转化为格式维度的numpy格式数组
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('./datasets/images/plane.jpg').resize((224, 224))  # 这里我们将图像resize为特定大小
x = transform_image(img)

target = 'llvm'

input_name = '0'  # 注意这里为之前导出onnx模型中的模型的输入id，这里为0
shape_dict = {input_name: x.shape}
# 利用Relay中的onnx前端读取我们导出的onnx模型
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

dtype = 'float32'
func = intrp.evaluate(sym)

output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
print(output.argmax())
