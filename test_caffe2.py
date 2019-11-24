import torch
import torchvision
import onnx
import time
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_runtime
from PIL import Image
import caffe2.python.onnx.backend


def main():
    model = torchvision.models.alexnet(pretrained=True)
    example = torch.rand(1, 3, 224, 224)
    torch_out = torch.onnx.export(model,
                                  example,
                                  "model.onnx",
                                  export_params=True
                                  )

    onnx_model = onnx.load('model.onnx')
    import pdb
    pdb.set_trace()
    #prepared_backend = caffe2.python.onnx.backend.prepare(model)
    print(1)


if __name__ == '__main__':
    main()
