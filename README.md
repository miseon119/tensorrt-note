# tensorrt-note

## ONNX to TensorRT

### Dynamic Shape ONNX to TensorRT Engine

```bash
trtexec --explicitBatch --onnx=yolo.onnx --saveEngine=model.plan --workspace=1024 --minShapes=input:1x3x608x608 --optShapes=input:2x3x608x608 --maxShapes=input:4x3x608x608 --fp16
```
> Note: You can change model.plan to model.engine; Your engine input name should match onnx file's input name.

Docker : `nvcr.io/nvidia/tensorrt:22.04-py3`

### Static Shape ONNX to TensorRT Engine

ONNX model in full-dimensions mode with **static input shapes**:
```bash
trtexec --onnx=model.onnx
```

> In this case, model.onnx has a static input shape, e.g. [8,3,416,416]


ONNX model with dynamic input `[-1,3,224,244]`
```bash
trtexec --onnx=model.onnx --shapes=input:32x3x244x244
```

[More](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

