# DeepSeek-V3 Local Setup Guide

> Setup guide and documentation for running DeepSeek-V3 locally using various inference frameworks

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DeepSeek-V3 can be deployed locally using the following hardware and open-source community software:

| Framework | Description |
|-----------|-------------|
| **DeepSeek-Infer Demo** | Simple and lightweight demo for FP8 and BF16 inference |
| **SGLang** | Fully supports DeepSeek-V3 in BF16 and FP8 inference modes |
| **LMDeploy** | Enables efficient FP8 and BF16 inference for local and cloud deployment |
| **TensorRT-LLM** | Currently supports BF16 inference and INT4/8 quantization |
| **vLLM** | Supports FP8 and BF16 modes for tensor parallelism and pipeline parallelism |
| **LightLLM** | Supports efficient single-node or multi-node deployment for FP8 and BF16 |
| **AMD GPU** | Runs DeepSeek-V3 on AMD GPUs via SGLang in BF16 and FP8 modes |
| **Huawei Ascend NPU** | Supports DeepSeek-V3 on Huawei Ascend devices in INT8 and BF16 |

> **Note:** Since FP8 training is natively adopted in our framework, we only provide FP8 weights. If you require BF16 weights for experimentation, use the provided conversion script:

```bash
cd inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

> **Note:** Hugging Face's Transformers has not been directly supported yet.

---

## 6.1 Inference with DeepSeek-Infer Demo (example only)

### System Requirements

> **Note:** Linux with Python 3.10 only. Mac and Windows are not supported.

### Dependencies

```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

### Model Weights & Demo Code Preparation

**Step 1:** Clone the DeepSeek-V3 GitHub repository:

```bash
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```

**Step 2:** Navigate to the inference folder and install dependencies:

```bash
cd DeepSeek-V3/inference
pip install -r requirements.txt
```

> Use a package manager like `conda` or `uv` to create a new virtual environment and install the dependencies.

**Step 3:** Download the model weights from Hugging Face, and put them into `/path/to/DeepSeek-V3` folder.

### Model Weights Conversion

Convert Hugging Face model weights to a specific format:

```bash
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

### Run

**Interactive chat with DeepSeek-V3:**

```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

**Batch inference on a given file:**

```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
```

---

## 6.2 Inference with SGLang (recommended)

SGLang currently supports MLA optimizations, DP Attention, FP8 (W8A8), FP8 KV Cache, and Torch Compile, delivering state-of-the-art latency and throughput performance among open-source frameworks.

- **SGLang v0.4.1** fully supports running DeepSeek-V3 on both **NVIDIA and AMD GPUs**
- Supports **multi-node tensor parallelism** for running on multiple network-connected machines
- **Multi-Token Prediction (MTP)** is in development

**Launch Instructions:** https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

---

## 6.3 Inference with LMDeploy (recommended)

LMDeploy is a flexible and high-performance inference and serving framework tailored for large language models. It offers offline pipeline processing, online deployment capabilities, and seamless integration with PyTorch-based workflows.

**Step-by-step instructions:** https://github.com/InternLM/lmdeploy/issues/2960

---

## 6.4 Inference with TRT-LLM (recommended)

TensorRT-LLM supports the DeepSeek-V3 model with the following precision options:

- **BF16**
- **INT4/INT8 weight-only**
- **FP8** (in progress, coming soon)

**Custom TRTLLM branch for DeepSeek-V3:** https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/deepseek_v3

---

## 6.5 Inference with vLLM (recommended)

**vLLM v0.6.6** supports DeepSeek-V3 inference for FP8 and BF16 modes on both NVIDIA and AMD GPUs. It also offers pipeline parallelism for running the model on multiple machines connected by networks.

**Detailed guidance:** https://github.com/vllm-project/vllm

---

## 6.6 Inference with LightLLM (recommended)

**LightLLM v1.0.1** supports single-machine and multi-machine tensor parallel deployment for DeepSeek-V3 (FP8/BF16) with mixed-precision deployment and more quantization modes continuously integrated.

**More details:** https://github.com/ModelTC/lightllm

---

## 6.7 Recommended Inference Functionality with AMD GPUs

In collaboration with the AMD team, Day-One support for AMD GPUs using SGLang has been achieved, with full compatibility for both FP8 and BF16 precision.

**Detailed guidance:** https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

---

## 6.8 Recommended Inference Functionality with Huawei Ascend NPUs

The MindIE framework from the Huawei Ascend community has successfully adapted the BF16 version of DeepSeek-V3.

**Step-by-step guidance for Ascend NPUs:** https://www.hiascend.com/en/

---

## 7. License

This code repository is licensed under the **MIT License**. The use of DeepSeek-V3 Base/Chat models is subject to the Model License. DeepSeek-V3 series (including Base and Chat) supports commercial use.

---

## 8. Citation

```bibtex
@misc{deepseekai2024deepseekv3technicalreport,
  title={DeepSeek-V3 Technical Report},
  author={DeepSeek-AI},
  year={2024},
  eprint={2412.19437},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.19437},
}
```

---

## 9. Contact

If you have any questions, please raise an issue or contact us at service@deepseek.com
