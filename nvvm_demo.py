import torch
import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image

from models.MobileNetV2 import mobilenet_v2


def main():
    model = mobilenet_v2(pretrained=True)
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

    print(1)
    img = Image.open('./datasets/images/plane.jpg').resize((224, 224))  # 这里我们将图像resize为特定大小
    # x = transform_image(img)
    # 以下的图片读取仅仅是为了测试
    img = np.array(img).transpose((2, 0, 1)).astype('float32')
    img = img / 255.0  # remember pytorch tensor is 0-1
    x = img[np.newaxis, :]
    print(2)
    import pdb
    pdb.set_trace()

    target = 'llvm'

    input_name = '0'  # 注意这里为之前导出onnx模型中的模型的输入id，这里为0
    shape_dict = {input_name: x.shape}
    # 利用Relay中的onnx前端读取我们导出的onnx模型
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    print(3)
    with relay.build_config(opt_level=3):
        intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

    print(4)
    dtype = 'float32'
    func = intrp.evaluate(sym)

    print(5)
    output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
    print(6)
    print(output.argmax())
    print(7)

    return

    # # 这里首先在PC的CPU上进行测试 所以使用LLVM进行导出
    # target = tvm.target.create('llvm')
    #
    # input_name = '0'  # change '1' to '0'
    # shape_dict = {input_name: x.shape}
    # sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    #
    # # 这里利用TVM构建出优化后模型的信息
    # with relay.build_config(opt_level=2):
    #     graph, lib, params = relay.build_module.build(sym, target, params=params)
    #
    # dtype = 'float32'

    # from tvm.contrib import graph_runtime
    #
    # # 下面的函数导出我们需要的动态链接库 地址可以自己定义
    # print("Output model files")
    # libpath = "../tvm_output_lib/mobilenet.so"
    # lib.export_library(libpath)
    #
    # # 下面的函数导出我们神经网络的结构，使用json文件保存
    # graph_json_path = "../tvm_output_lib/mobilenet.json"
    # with open(graph_json_path, 'w') as fo:
    #     fo.write(graph)
    #
    # # 下面的函数中我们导出神经网络模型的权重参数
    # param_path = "../tvm_output_lib/mobilenet.params"
    # with open(param_path, 'wb') as fo:
    #     fo.write(relay.save_param_dict(params))
    # # -------------至此导出模型阶段已经结束--------
    #
    # # 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
    # loaded_json = open(graph_json_path).read()
    # loaded_lib = tvm.module.load(libpath)
    # loaded_params = bytearray(open(param_path, "rb").read())
    #
    # # 这里执行的平台为CPU
    # ctx = tvm.cpu()
    #
    # module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    # module.load_params(loaded_params)
    # module.set_input("0", x)
    # module.run()
    # out_deploy = module.get_output(0).asnumpy()
    #
    # print(out_deploy)


if __name__ == '__main__':
    main()