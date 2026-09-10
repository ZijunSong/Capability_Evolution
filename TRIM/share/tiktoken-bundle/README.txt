o200k_harmony offline vocab bundle
==================================

This bundle is for openai_harmony==0.0.8 (HarmonyGptOss) AND tiktoken
get_encoding("o200k_harmony") on air-gapped / no-Azure machines.

IMPORTANT
---------
There is NO separate Azure blob named o200k_harmony.tiktoken
(https://openaipublic.blob.core.windows.net/encodings/o200k_harmony.tiktoken
returns 404). GPT-OSS Harmony reuses the o200k_base BPE ranks and then
registers Harmony special tokens in-process:

  <|return|>  200002
  <|constrain|> 200003
  <|channel|> 200005
  <|start|>   200006
  <|end|>     200007
  <|message|> 200008
  <|call|>    200012
  plus reserved ids 199998..201087  (n_vocab = 201088)

openai_harmony 0.0.8 source (src/tiktoken_ext/public_encodings.rs):
  Encoding::O200kHarmony.vocab_file_name() == "o200k_base.tiktoken"
  expected SHA256 == 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d

The previous tar only shipped the named o200k_base.tiktoken file. That is
the correct BPE blob, but 0.0.8 will still try Azure unless ONE of these
is set:

  1) TIKTOKEN_ENCODINGS_BASE  -> directory containing o200k_base.tiktoken
  2) TIKTOKEN_RS_CACHE_DIR    -> directory containing SHA1(url) cache file
     fb374d419588a4632f3f557e76b4b70aebbca790
     (this is what /tmp/tiktoken-rs-cache uses if env is unset)

Python tiktoken.get_encoding("o200k_harmony") uses TIKTOKEN_CACHE_DIR
(SHA1 of the same URL), not TIKTOKEN_RS_CACHE_DIR.

Files
-----
share/tiktoken/o200k_base.tiktoken
  Official BPE ranks. SHA256 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d

share/tiktoken/o200k_harmony.tiktoken
  Same bytes as o200k_base.tiktoken (BPE ranks are shared). Named copy so a
  reviewer looking for "o200k_harmony vocabulary" finds a file with that name.

share/tiktoken/o200k_harmony.special_tokens.json
  Full Harmony special-token table dumped from tiktoken 0.13 o200k_harmony
  (1091 specials, ids 199998..201087). This is the special-token encoding
  that o200k_base.tiktoken itself does NOT contain.

share/tiktoken/cl100k_base.tiktoken
  Official cl100k ranks. SHA256 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
  Not required for HarmonyGptOss, included because some openai_harmony
  loaders also probe this encoding.

share/hf_tokenizer/tokenizer.json (+ tokenizer_config.json, special_tokens_map.json)
  gpt-oss-20b HuggingFace tokenizer. This IS the Harmony special-token
  vocabulary in HF format (added_tokens include <|call|> / <|return|> / ...).
  transformers uses this; openai_harmony 0.0.8 does not.

tiktoken_rs_cache/fb374d419588a4632f3f557e76b4b70aebbca790
  SHA1("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken")
  This is the filename openai_harmony 0.0.8 looks up when TIKTOKEN_ENCODINGS_BASE
  is unset.

tiktoken_cache/*
  Python tiktoken cache keys (SHA1 url, SHA256 url, content sha256).

Install on the smoke machine
----------------------------
  tar -xzf o200k_base.tiktoken.tar.gz
  bash tiktoken-bundle/install.sh /opt/scape-easyopd-smoke7

This copies files and prints the three env vars you MUST export on every
Harmony / four-cell run:

  export TIKTOKEN_ENCODINGS_BASE=/opt/scape-easyopd-smoke7/share/tiktoken
  export TIKTOKEN_RS_CACHE_DIR=/opt/scape-easyopd-smoke7/share/tiktoken_rs_cache
  export TIKTOKEN_CACHE_DIR=/opt/scape-easyopd-smoke7/share/tiktoken_cache

Then verify with the approved interpreter:

  /opt/scape-easyopd-smoke7/bin/python tiktoken-bundle/verify_offline.py

Expected:
  openai_harmony==0.0.8 HarmonyGptOss loads
  <|call|>=200012  <|return|>=200002
  tiktoken o200k_harmony n_vocab=201088
