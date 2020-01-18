import torch
import torchvision
import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_runtime
from PIL import Image


def main():
    # 加载onnx模型
    model = onnx.load("lstm.onnx")

    # 利用Relay中的onnx前端读取我们导出的onnx模型
    sym, params = relay.frontend.from_onnx(model, {'inputs': (28, 128, 28)})

    # OPT_PASS_LEVEL = {
    #     "SimplifyInference": 0,
    #     "OpFusion": 1,
    #     "FoldConstant": 2,
    #     "CombineParallelConv2D": 3,
    #     "FoldScaleAxis": 3,
    #     "AlterOpLayout": 3,
    #     "CanonicalizeOps": 3,
    # }
    target = 'llvm'
    with relay.build_config(opt_level=3):
        intrp = relay.build_module.create_executor('graph', sym, tvm.cpu(0), target)

    func = intrp.evaluate(sym)
    inputs = np.ones([28, 128, 28]).float()
    output = func(tvm.nd.array(inputs.astype('float32')), **params).asnumpy()
    print(output.argmax())

    # 这里利用TVM构建出优化后模型的信息
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(sym, target, params=params)


    # 下面的函数导出我们需要的动态链接库 地址可以自己定义
    print("Output model files")
    libpath = "./model.so"
    lib.export_library(libpath)

    # 下面的函数导出我们神经网络的结构，使用json文件保存
    graph_json_path = "./model.json"
    with open(graph_json_path, 'w') as fo:
        fo.write(graph)

    # 下面的函数中我们导出神经网络模型的权重参数
    param_path = "./model.params"
    with open(param_path, 'wb') as fo:
        fo.write(relay.save_param_dict(params))
    # -------------至此导出模型阶段已经结束--------

    # 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
    loaded_json = open(graph_json_path).read()
    loaded_lib = tvm.module.load(libpath)
    loaded_params = bytearray(open(param_path, "rb").read())

    # 这里执行的平台为CPU
    ctx = tvm.cpu()

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("0", inputs)
    module.run()
    out_deploy = module.get_output(0).asnumpy()

    print(out_deploy)


if __name__ == '__main__':
    main()
