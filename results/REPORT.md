# MobileNetV2 GPU/CPU benchmark — MNN vs TFLite (LiteRT)

**Device:** Samsung Galaxy Z Fold7 (SM-F766B, Exynos 2500, Xclipse 950 GPU)  
**ABI:** arm64-v8a · Android 16 · NDK 27.0.12077973  
**Model:** MobileNetV2 (1.0_224, fp32, ImageNet weights)  
**Input:** seed=42 uniform_real(-1, 1) generated in C++ from `std::mt19937_64`.

Each backend was compared to a reference output dumped from a previous CPU-FP32 run of the SAME framework — TFLite XNNPACK CPU for the TFLite side, MNN CPU for the MNN side.

- Warmup runs: 5 · Timed runs: 30
- Each MNN GPU run starts cold (cache deleted before run).

## Headline numbers (all times in milliseconds)

| Framework | Backend | Variant | Create | First inf. | Warm mean | Inf. p50 | Inf. p90 | Inf. mean ± stdev | Cosine vs ref | Argmax match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| tflite | xnnpack | - | 20.7 | 22.8 | 9.31 | 3.68 | 4.80 | 3.96 ± 0.60 | - | - |
| tflite | cpu | fp32 | 4.7 | 87.9 | 91.98 | 115.94 | 126.46 | 112.68 ± 12.37 | 1.0000 | ✓ |
| tflite | gpu | fp16 | 1212.6 | 14.0 | 12.90 | 4.37 | 7.90 | 5.11 ± 1.80 | 0.9998 | ✓ |
| tflite | gpu | fp32 | 1312.6 | 17.8 | 13.74 | 5.47 | 7.59 | 5.94 ± 1.27 | 1.0000 | ✓ |
| tflite | gpu | cl | 1260.5 | 14.8 | 12.21 | 4.42 | 7.12 | 5.12 ± 1.78 | 0.9998 | ✓ |
| mnn | cpu | fp32 | 21.8 | 21.7 | 10.21 | 6.13 | 8.12 | 6.36 ± 1.18 | - | - |
| mnn | opencl | buffer | 251.4 | 2716.8 | 16.15 | 5.89 | 7.66 | 6.06 ± 1.49 | 0.9999 | ✓ |
| mnn | opencl | image | 381.4 | 1681.7 | 11.06 | 6.96 | 11.95 | 7.64 ± 2.58 | 0.9999 | ✓ |
| mnn | opencl | fp32 | 252.2 | 2784.1 | 18.97 | 6.70 | 8.89 | 6.43 ± 1.42 | 1.0000 | ✓ |
| mnn | vulkan | fp16 | 302.4 | 5486.6 | 9.97 | 11.26 | 13.97 | 11.09 ± 2.49 | 0.9998 | ✓ |
| mnn | vulkan | fp32 | 331.5 | 5780.4 | 11.77 | 13.70 | 15.81 | 14.04 ± 1.27 | 0.9998 | ✓ |

## CPU comparison (4 threads, fp32)

| Path | Inf. p50 | Inf. mean | Notes |
|---|---:|---:|---|
| tflite_xnnpack_fp32 | 3.68 | 3.96 | TFLite + XNNPACK delegate (TFLite mobile production default). |
| tflite_cpu_fp32 | 115.94 | 112.68 | TFLite C API w/o delegate (basic kernels — not used in production). |
| mnn_cpu_fp32 | 6.13 | 6.36 | MNN CPU (NEON / SVE / i8mm / fp16-dot). |

## GPU comparison (Xclipse 950)

| Path | Create | First inf. | Inf. p50 | Inf. mean | Cosine |
|---|---:|---:|---:|---:|---:|
| tflite_gpu_fp16 | 1212.6 | 14.0 | 4.37 | 5.11 | 0.9998 |
| tflite_gpu_fp32 | 1312.6 | 17.8 | 5.47 | 5.94 | 1.0000 |
| tflite_gpu_cl_fp16 | 1260.5 | 14.8 | 4.42 | 5.12 | 0.9998 |
| mnn_opencl_buffer_fp16 | 251.4 | 2716.8 | 5.89 | 6.06 | 0.9999 |
| mnn_opencl_image_fp16 | 381.4 | 1681.7 | 6.96 | 7.64 | 0.9999 |
| mnn_opencl_buffer_fp32 | 252.2 | 2784.1 | 6.70 | 6.43 | 1.0000 |
| mnn_vulkan_fp16 | 302.4 | 5486.6 | 11.26 | 11.09 | 0.9998 |
| mnn_vulkan_fp32 | 331.5 | 5780.4 | 13.70 | 14.04 | 0.9998 |

## Cold-start cost (only GPU paths have JIT-time)

MNN does kernel JIT on **first inference**; TFLite does it during **create**. The cold totals are similar but the latency is split very differently.

| Path | Create | First inf. | Cold total |
|---|---:|---:|---:|
| tflite_gpu_fp16 | 1212.6 | 14.0 | **1226.6** |
| tflite_gpu_fp32 | 1312.6 | 17.8 | **1330.4** |
| tflite_gpu_cl_fp16 | 1260.5 | 14.8 | **1275.3** |
| mnn_opencl_buffer_fp16 | 251.4 | 2716.8 | **2968.2** |
| mnn_opencl_image_fp16 | 381.4 | 1681.7 | **2063.0** |
| mnn_opencl_buffer_fp32 | 252.2 | 2784.1 | **3036.2** |
| mnn_vulkan_fp16 | 302.4 | 5486.6 | **5789.1** |
| mnn_vulkan_fp32 | 331.5 | 5780.4 | **6111.9** |

## Accuracy vs same-framework CPU-FP32 reference

| Path | Cosine | Max abs | Mean abs | Rel. L2 | Argmax (got / ref) |
|---|---:|---:|---:|---:|:---:|
| tflite_cpu_fp32 | 1.000000 | 0.0000 | 0.00000 | 0.00000 | 885 / 885 |
| tflite_gpu_fp16 | 0.999763 | 0.1520 | 0.02768 | 0.02194 | 885 / 885 |
| tflite_gpu_fp32 | 1.000000 | 0.0000 | 0.00000 | 0.00000 | 885 / 885 |
| tflite_gpu_cl_fp16 | 0.999763 | 0.1520 | 0.02768 | 0.02194 | 885 / 885 |
| mnn_opencl_buffer_fp16 | 0.999901 | 0.1149 | 0.02313 | 0.01458 | 446 / 446 |
| mnn_opencl_image_fp16 | 0.999923 | 0.0766 | 0.01955 | 0.01242 | 446 / 446 |
| mnn_opencl_buffer_fp32 | 0.999971 | 0.0677 | 0.01185 | 0.00761 | 446 / 446 |
| mnn_vulkan_fp16 | 0.999830 | 0.1342 | 0.02983 | 0.01900 | 446 / 446 |
| mnn_vulkan_fp32 | 0.999830 | 0.1342 | 0.02983 | 0.01900 | 446 / 446 |

## Best configuration per framework

- **Fastest CPU overall**: `tflite_xnnpack_fp32` — p50 3.68 ms.
- **Fastest GPU overall**: `tflite_gpu_fp16` — p50 4.37 ms.
- **Best MNN GPU config**: `mnn_opencl_buffer_fp16` — p50 5.89 ms.
- **Best TFLite GPU config**: `tflite_gpu_fp16` — p50 4.37 ms.

