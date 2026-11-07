# GreyParrot-v4-64M

分块扫描、融合训练算子与 RKM/TSA 缓存。仓库发布名为 GreyParrot-v4-64M，代码内部架构版本为 v6。

## 使用

```python
import torch
from model_v6_fast import GreyParrotV6, build_64m

# 直接加载原 v6 checkpoint，权重名和形状不变。
model = GreyParrotV6(V=32018, cfg=cfg)
model.load_state_dict(checkpoint["model"])
model.eval()

out = model.generate(prompt_ids, max_new=100, use_cache=True,
                     temperature=0.8, top_p=0.9)
```

`use_cache=False` 可回退到原始整段前向。训练仍调用 `model(x, y)`。Fast 版本使用自定义 `FusedTeLUGLU`／`FusedRMSNorm` autograd 节点，RKM 使用数值等价的两级分块仿射扫描；默认扫描块长为 64，可用配置项 `rkm_scan_chunk` 或训练参数 `--rkm-scan-chunk` 调整。

## 实现内容

- RKM 两级分块扫描：块内 inclusive 仿射扫描、块摘要扫描、prefix state 合并；支持非整除尾块和非零初始状态。短于等于块长时直接走原扫描，模型 8 层逐层循环保持不变。
- TSA 自适应 query 分块：按元素预算自动选择 1～3 块，不再随预算下降产生任意长的 Python 循环；显式 `--tsa-chunk` 仍可覆盖自动模式。
- TeLU-GLU 自定义 autograd：一次 forward 节点和一次 backward 节点，同时计算 gate/up 梯度；减少 autograd 图和中间保存。它是算子图合并，不等于 eager 模式已经生成单个硬件 kernel。
- 减少同步：CUDA 梯度裁剪优先使用 fused 实现；生成时先满足 `min_len` 再检查 EOS，避免早期每步 GPU→CPU 同步。
- TSA 缓存：逐层预分配 K/V、query、方向分数和独立 query/key 路由投影；每个完成块只计算一次摘要、方向地标和压缩 KV。
- RKM 缓存：旋转坐标递归状态、累计相位和梯形规则的原始前一项输入。
- 严格因果：当前 token 刚完成的块不会立即被自身读取，下一 token 才可读取。
- RoPE 使用绝对位置 offset；当前支持默认 `rkm_conv_k=1`。

## 验证结果

- 小模型融合损失误差：0；参数梯度最大绝对误差：6.33e-8。
- 分块扫描覆盖 T=64/65/129/130/512、非整除尾块、零／非零初始状态：最大输出误差 1.78e-15，最大输入梯度误差 1.74e-18（FP64 单元测试）。
- 整模型 T=130 分块路径：损失误差 0，参数梯度最大误差 3.73e-8，prefill 最大误差 1.56e-7。
- TSA 自动 query 循环已实测触发 1、2、3 块，并在极低预算下封顶为 3。
- 缓存测试覆盖 12 个逐步 token，并跨过完成块边界。
- 8.20M 已训练检查点：贪心缓存／整段序列一致；逐步 logits 最大差约 0.038。
- 完整 64M：8 个贪心生成 token 与整段前向一致，最大 logits 差 0.0359。
- 随机采样不保证序列完全相同：接近均匀或并列概率会因单 token GEMM 与整段 GEMM 的舍入差而改变抽样。

## 当前双核 CPU 实测

完整 64M、prompt 64、生成 8 tokens：

- 无缓存：1.3279 秒。
- 有缓存：0.3914 秒。
- 加速：3.39 倍。
- 峰值进程内存：0.864 GB。

8.20M、prompt 24、生成 30 tokens：缓存约 1.56 倍。短生成的 prefill 和 Python 算子开销占比较高；更长生成通常收益更明显。以上不是 W7900D 数据。

## 训练加速实测

完整 64M，双核 CPU，batch 1、序列 512、FP32：

| 实现 | 前向 | 反向 | 合计 |
|---|---:|---:|---:|
| 原 v6，CE chunk 256 | 2.903 s | 38.778 s | 41.681 s |
| Fast，CE chunk 256 | 2.624 s | 24.456 s | 27.080 s |
| Fast 分块扫描前，CE chunk 512 | 2.490 s | 22.685 s | 25.175 s |
| Fast＋RKM chunk 64，CE chunk 512，第 1 次 | 2.643 s | 2.721 s | **5.363 s** |
| 同上，独立进程复测 | 2.557 s | 2.824 s | **5.381 s** |

分块扫描版相对原版整步约加速 7.75 倍，相对分块扫描前的 Fast 版约加速 4.68 倍；反向约加速 8.11 倍。以上为当前沙箱双核 CPU 的两次独立进程实测，不是 W7900D 数据；GPU 上需重新选择扫描块长并实测。梯度等价验证通过。GPU 推荐从 `--ce-chunk 1024 --rkm-scan-chunk 64` 开始，不默认开激活检查点。

```bash
python train_v6_fast.py \
  --data train_tokens.int32.npy \
  --val-data validation_tokens.int32.npy \
  --out runs/gp64 --device cuda --dtype bf16 \
  --batch-size 8 --seq-len 512 --ce-chunk 1024 \
  --rkm-scan-chunk 64 --grad-accum 1 --max-steps 10000
```

新增 `--max-seconds` 限时基准、`--compile-mode` 编译策略、独立验证集与估算训练 TFLOPS 日志。限时停止也会保存检查点。

## 后续 GPU 优化边界

当前版本尚未使用 Triton／HIP 写真正的单 kernel RMSNorm、RoPE、TeLU-GLU 或 MTA。ROCm 上建议先对 Fast 模型做 BF16 和 `torch.compile` 实测，再决定是否写专用 kernel；不能把自定义 autograd 节点的收益冒充为硬件融合收益。
