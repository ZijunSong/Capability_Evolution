# Tiktoken / Harmony 词表离线搜索结论（2026-09-10）

## 搜索结论

1. **目标机器上未发现**独立的 `o200k_harmony.tiktoken`、`o200k_base.tiktoken`、`cl100k_base.tiktoken` 文件，也未配置 `TIKTOKEN_ENCODINGS_BASE` / `TIKTOKEN_CACHE_DIR`。
2. **GPT-OSS / harness-1 的 `tokenizer.json` 存在且结构完整**，但 `openai_harmony==0.0.8` 的 `HarmonyGptOss` **不会**读取 HF tokenizer；它通过 tiktoken-rs 加载 **`o200k_base.tiktoken`** 并在进程内注册 Harmony special tokens。
3. **Azure 上不存在**名为 `o200k_harmony.tiktoken` 的独立 blob（404）。`o200k_harmony` 编码 = `o200k_base` BPE ranks + 1091 个 Harmony special tokens（见 `o200k_harmony.special_tokens.json`）。

## 本 bundle 内容

| 路径 | 用途 |
|---|---|
| `share/tiktoken/o200k_base.tiktoken` | openai_harmony / tiktoken 离线 BPE 词表（必需） |
| `share/tiktoken/o200k_harmony.tiktoken` | 与 o200k_base 同内容的命名副本 |
| `share/tiktoken/o200k_harmony.special_tokens.json` | Harmony special token 表（文档/审计用） |
| `share/tiktoken/cl100k_base.tiktoken` | 可选；部分工具链会探测 cl100k |
| `tiktoken_rs_cache/` | openai_harmony Rust 侧 SHA1 缓存 |
| `tiktoken_cache/` | Python tiktoken SHA1/SHA256 缓存 |
| `share/hf_tokenizer/` | gpt-oss HF tokenizer（**不能**替代 HarmonyGptOss，仅供 transformers 对照） |

## 新服务器用法

```bash
export TIKTOKEN_ENCODINGS_BASE=/path/to/TRIM/share/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/path/to/TRIM/share/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/path/to/TRIM/share/tiktoken-bundle/tiktoken_cache

python TRIM/share/tiktoken-bundle/verify_offline.py
```

或解压 tar 后运行 `bash install.sh /opt/your-prefix`。

## 验证

`verify_offline.py` 已通过离线检查：`HarmonyGptOss` 与 `tiktoken.get_encoding("o200k_harmony")` 均能正确解析 `<|call|>`=200012、`<|return|>`=200002 等 special tokens。
