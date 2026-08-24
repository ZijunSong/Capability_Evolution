from vllm import LLM, SamplingParams


def main() -> None:
    model = "/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"
    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        tensor_parallel_size=8,
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
    )
    out = llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))
    print("PROBE_OK", len(out), out[0].outputs[0].text, flush=True)


if __name__ == "__main__":
    main()
