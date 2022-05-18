# tensorrt-notes

## ONNX to TensorRT

### Dynamic Shape ONNX to TensorRT Engine

```bash
trtexec --explicitBatch --onnx=yolo.onnx --saveEngine=model.plan --workspace=1024 --minShapes=input:1x3x608x608 --optShapes=input:2x3x608x608 --maxShapes=input:4x3x608x608 --fp16
```
> Note: You can change model.plan to model.engine; Your engine input name should match onnx file's input name.

Docker : `nvcr.io/nvidia/tensorrt:21.10-py3`
