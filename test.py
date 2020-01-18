import torch
import torchvision
from torch.nn import Linear, Sequential, Module

import onnx
import tvm
import tvm.relay as relay


def test():
    class Flatten(Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    file_name = "./test.onnx"
    input_size = (1, 16, 32, 32)
    dummy_input = torch.randn(*input_size)
    layer = Sequential(Flatten(), Linear(16 * 32 * 32, 64))
    torch.onnx.export(layer, dummy_input, file_name, export_params=True)

    onnx_model = onnx.load(file_name)
    relay.frontend.from_onnx(onnx_model, {'0': input_size})

    from torch.autograd import Variable
    import torch.onnx
    import torchvision

    dummy_input = Variable(torch.randn(10, 3, 224, 224))
    model = torchvision.models.alexnet(pretrained=True)
    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True)
    model = onnx.load("alexnet.onnx")
    relay.frontend.from_onnx(model, {'input.1': (10, 3, 224, 224)})


def main():
    model = onnx.load("alexnet.onnx")
    relay.frontend.from_onnx(model, {'input.1': (28, 128, 28)})
    print(1)


if __name__ == '__main__':
    main()
