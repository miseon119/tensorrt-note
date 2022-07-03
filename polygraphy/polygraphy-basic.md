# Polygraphy Basic

## polygraphy run

> Example 1: 用上面的 .onnx 构建一个 TensorRT 引擎，使用 FP16精度，同时在 onnxruntime 和 TensorRT 中运行，对比结果

```bash
polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP16.plan \
    --atol 1e-3 --rtol 1e-3 \
    --fp16 \
    --verbose \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-run-FP16.txt
```

```
# 注意参数名和格式跟 trtexec 不一样，多个形状之间用空格分隔，如：
# --trt-max-shapes 'input0:[16,320,256]' 'input1:[16，320]' 'input2:[16]'
```

> Example 2:  .onnx 构建一个 TensorRT 引擎，输出所有层的计算结果作对比

```bash
polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP32-MarkAll.plan \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    > result-run-FP32-MarkAll.txt
```
part of output:
![polygraphy1](./images/polygraphy1.png) 