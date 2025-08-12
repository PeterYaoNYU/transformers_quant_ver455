# eval_gsm8k_single_gpu_batched.py
import os, re, math, json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, HQQQuantizedCache
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-8B"
USE_THINKING = True  # set True only if your checkpoint supports thinking mode

# ---- batching ----
BATCH_SIZE = 32  # tweak to taste

# ---- quantized KV cache flags ----
USE_QUANT_CACHE = True
QUANT_BACKEND   = "HQQ"   # "HQQ" or "quanto"
N_BITS          = 4
AXIS_KEY        = 1 if QUANT_BACKEND == "HQQ" else 0   # HQQ often uses 1; quanto uses {0,-1}
AXIS_VALUE      = 1 if QUANT_BACKEND == "HQQ" else 0
Q_GROUP_SIZE    = 64
RESIDUAL_LEN    = 128

# ---------- regex helpers ----------
# _PAT_BOX  = re.compile(r"\\+boxed\s*\{\s*([^}]*)\s*\}")
# _PAT_HASH = re.compile(r"####\s*([^\n]+)")
# _PAT_NUM  = re.compile(r"-?\d+(?:\.\d+)?")

_PAT_BOX  = re.compile(r"\\boxed\s*\{\s*([^{}]*)\s*\}")
_PAT_HASH = re.compile(r"####\s*([^\n]+)")
_PAT_NUM  = re.compile(r"-?\d+(?:\.\d+)?")

_INSTR_EMPTY_BOX = re.compile(
    r"(?is)please\s+reason[\s\S]*?final\s+answer\s+within\s*\\boxed\s*\{\s*\}\.?"
)
_INSTR_EMPTY_BOX_SHORT = re.compile(
    r"(?is)final\s+answer\s+within\s*\\boxed\s*\{\s*\}\.?"
)

def _preclean(t: str) -> str:
    # remove the instructional line that contains the empty \boxed{ }
    t = _INSTR_EMPTY_BOX.sub("", t)
    t = _INSTR_EMPTY_BOX_SHORT.sub("", t)
    return t

def extract_final(t: str) -> str:
    t = _preclean(t)
    # 1) get all boxed matches
    boxes = [m.group(1).strip() for m in _PAT_BOX.finditer(t)]

    # 2) try the last boxed that has a number
    for s in reversed(boxes):
        nums = _PAT_NUM.findall(s)
        if nums:
            return nums[-1]

    # 3) otherwise last non-empty boxed
    for s in reversed(boxes):
        if s:
            return s

    # 4) fallbacks: #### answer
    m = _PAT_HASH.search(t)
    if m:
        s = m.group(1).strip()
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s)

    # 5) fallback: Final Answer:
    m = re.search(r"(?i)final\s*answer[:\s-]*([^\n]+)", t)
    if m:
        s = m.group(1).strip()
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s)

    # 6) fallback: last number in the text
    nums = _PAT_NUM.findall(t)
    return (nums[-1] if nums else t.strip())

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
        torch.cuda.set_device(device.index or 0)
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
    ds = load_dataset("openai/gsm8k", "main", split="test[:128]")

    # ---- generation kwargs ----
    gen_kwargs = dict(
        do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=3500, pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # pick cache class based on backend
    CacheClass = HQQQuantizedCache if QUANT_BACKEND.upper() == "HQQ" else QuantoQuantizedCache
    kv_args = dict(
        nbits=N_BITS,
        axis_key=AXIS_KEY,
        axis_value=AXIS_VALUE,
        q_group_size=Q_GROUP_SIZE,
        residual_length=RESIDUAL_LEN,
    )

    # ---- eval loop (batched) ----
    out_dir = "sg_out"
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"gsm8k_{MODEL_ID.replace('/','_')}_batched.jsonl"

    correct = 0
    records = []

    with open(out_path, "w", encoding="utf-8") as fout, torch.no_grad():
        for start in tqdm(range(0, len(ds), BATCH_SIZE)):
            end = min(len(ds), start + BATCH_SIZE)
            idxs = list(range(start, end))

            questions = [ds[i]["question"] for i in idxs]
            golds     = [extract_gold(ds[i]["answer"]) for i in idxs]
            prompts   = [build_prompt(tokenizer, q) for q in questions]

            toks = tokenizer(prompts, return_tensors="pt", padding=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            
            print("input_ids:", toks["input_ids"].shape, "attn:", toks["attention_mask"].shape)

            # Fresh quantized cache per batch
            past_key_values = CacheClass(config=model.config, **kv_args)

            out = model.generate(**toks, **gen_kwargs, past_key_values=past_key_values)  # [B, out_len]

            # decode per row, slicing off each row's true prompt length
            attn = toks["attention_mask"]  # [B, Tin]
            input_lens = attn.sum(dim=1).tolist()

            for row, (i, gold, in_len) in enumerate(zip(idxs, golds, input_lens)):
                gen_ids = out[row, in_len:]  # continuation only for this row
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
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
                rec = {
                    "idx": i,
                    "question": ds[i]["question"],
                    "pred_raw": text,
                    "pred": pred,
                    "gold": gold,
                    "correct": bool(ok),
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = correct / len(ds)
    merged = f"gsm8k_{MODEL_ID.replace('/','_')}_batched.merged.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Batched] Accuracy on first {len(ds)} GSM8K: {acc:.3f}  ({correct}/{len(ds)})")
    print(f"Saved details to {merged}")

if __name__ == "__main__":
    main()
