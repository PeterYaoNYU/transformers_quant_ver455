# eval_gsm8k_single_gpu.py
import os, re, math, json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, HQQQuantizedCache

from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-32B"
USE_THINKING = True  # set True only if your checkpoint supports thinking mode

# ---- quantized KV cache flags ----
USE_QUANT_CACHE = True
QUANT_BACKEND   = "HQQ"   # "HQQ" or "quanto"
N_BITS          = 4
AXIS_KEY        = 1 if QUANT_BACKEND == "HQQ" else 0   # common HQQ choice is 1; quanto uses {0,-1}
AXIS_VALUE      = 1 if QUANT_BACKEND == "HQQ" else 0
Q_GROUP_SIZE    = 64
RESIDUAL_LEN    = 128

# ---------- regex helpers ----------
# (slightly more tolerant boxed regex)
_PAT_BOX  = re.compile(r"\\+boxed\s*\{\s*([^}]*)\s*\}")
_PAT_HASH = re.compile(r"####\s*([^\n]+)")
_PAT_NUM  = re.compile(r"-?\d+(?:\.\d+)?")

def extract_final(text: str) -> str:
    # Strip hidden thoughts if present (avoid blanks when skip_special_tokens hid content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL|re.IGNORECASE)

    m = _PAT_BOX.search(text)
    if m:
        s = m.group(1)
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s).strip()

    m = _PAT_HASH.search(text)
    if m:
        s = m.group(1)
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s).strip()

    m = re.search(r"(?i)final\s*answer[:\s-]*([^\n]+)", text)
    if m:
        s = m.group(1)
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s).strip()

    nums = _PAT_NUM.findall(text)
    return (nums[-1].strip() if nums else text.strip())

def extract_gold(answer: str) -> str:
    m = _PAT_HASH.search(answer)
    if m:
        nums = _PAT_NUM.findall(m.group(1))
        return nums[-1] if nums else m.group(1).strip()
    nums = _PAT_NUM.findall(answer)
    return nums[-1] if nums else answer.strip()

def build_prompt(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q.strip() +
             "\n\nPlease reason step by step, and put your final answer within \\boxed{ }."}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=USE_THINKING
    )

def main():
    # ---- device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ---- tokenizer/model ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    # ---- data ----
    ds = load_dataset("openai/gsm8k", "main", split="test[:15]")
    # past_key_values = QuantoQuantizedCache(config=model.config, nbits=4)

    # ---- generation kwargs (preserving quantized KV cache) ----
    gen_kwargs = dict(
        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        max_new_tokens=3500, pad_token_id=tokenizer.eos_token_id,
        # cache_implementation="quantized",
        # cache_config={"backend": "HQQ", "nbits": 4},  # HQQ + bs=1
    )

    if USE_QUANT_CACHE:
    #     # This matches newer Transformers that expect cache_implementation + cache_config with a backend
        gen_kwargs.update(
            # cache_implementation="quantized",
            use_cache = True,
            # cache_config={
            #     # "backend": QUANT_BACKEND,     # "HQQ" or "quanto"
            #     "nbits": N_BITS,
            #     "axis_key": AXIS_KEY,
            #     "axis_value": AXIS_VALUE,
            #     "q_group_size": Q_GROUP_SIZE,
            #     "residual_length": RESIDUAL_LEN,
            #     # "compute_dtype": torch.float16,  # optional in some branches
            #     "device": str(device),           # optional; usually inferred
            # },
        )

    # ---- eval loop (single GPU, bs=1) ----
    out_dir = "sg_out"
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"gsm8k_{MODEL_ID.replace('/','_')}_single_gpu.jsonl"

    correct = 0
    records = []

    with open(out_path, "w", encoding="utf-8") as fout, torch.no_grad():
        for i in tqdm(range(len(ds))):
            q = ds[i]["question"]
            gold = extract_gold(ds[i]["answer"])
            prompt = build_prompt(tokenizer, q)

            toks = tokenizer(prompt, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}

            past_key_values = HQQQuantizedCache(config=model.config, nbits=4, axis_key=1, axis_value=1)


            out = model.generate(**toks, **gen_kwargs, past_key_values=past_key_values)

            # Keep specials, then strip <think> manually to avoid blank preds
            gen_ids = out[0, toks["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred = extract_final(text)

            is_num = _PAT_NUM.fullmatch(pred.strip()) and _PAT_NUM.fullmatch(gold.strip())
            if is_num:
                try:
                    ok = math.isclose(float(pred), float(gold), rel_tol=0, abs_tol=1e-6)
                except Exception:
                    ok = (pred.strip() == gold.strip())
            else:
                ok = (pred.strip() == gold.strip())

            correct += int(ok)
            rec = {"idx": i, "question": q, "pred_raw": text, "pred": pred, "gold": gold, "correct": bool(ok)}
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = correct / len(ds)
    merged = f"gsm8k_{MODEL_ID.replace('/','_')}_single_gpu.merged.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Single GPU] Accuracy on first 100 GSM8K: {acc:.3f}  ({correct}/{len(ds)})")
    print(f"Saved details to {merged}")

if __name__ == "__main__":
    main()
