# eval_gsm8k_dp.py
import os, re, math, json, math as _math
from pathlib import Path
import multiprocessing as mp

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, HQQQuantizedCache
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-8B"
USE_THINKING = True

# ---- batching ----
BATCH_SIZE = 32

# ---- quantized KV cache flags ----
USE_QUANT_CACHE = False
QUANT_BACKEND   = "HQQ"   # "HQQ" or "quanto"
N_BITS          = 4
AXIS_KEY        = 1 if QUANT_BACKEND == "HQQ" else 0
AXIS_VALUE      = 1 if QUANT_BACKEND == "HQQ" else 0
Q_GROUP_SIZE    = 64
RESIDUAL_LEN    = 128

# ---------- regex helpers ----------
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
    
    boxes = [m.group(1).strip() for m in _PAT_BOX.finditer(t)]
    for s in reversed(boxes):
        nums = _PAT_NUM.findall(s)
        if nums:
            return nums[-1]
    for s in reversed(boxes):
        if s:
            return s
    m = _PAT_HASH.search(t)
    if m:
        s = m.group(1).strip()
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s)
    m = re.search(r"(?i)final\s*answer[:\s-]*([^\n]+)", t)
    if m:
        s = m.group(1).strip()
        nums = _PAT_NUM.findall(s)
        return (nums[-1] if nums else s)
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

def run_worker(rank: int, world_size: int, subset_spec: str, out_dir: str):
    # Optional: reduce CPU contention when you spawn multiple procs
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Prefer FlashAttention/SDPA variants
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    ds = load_dataset("openai/gsm8k", "main", split=subset_spec)
    N = len(ds)

    # Shard dataset into contiguous chunks for better padding efficiency
    per_rank = (N + world_size - 1) // world_size
    start = rank * per_rank
    end   = min(N, (rank + 1) * per_rank)
    if start >= end:
        return  # this rank has nothing

    # Generation kwargs (you can tune)
    gen_kwargs = dict(
        do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=3500,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    CacheClass = HQQQuantizedCache if QUANT_BACKEND.upper() == "HQQ" else QuantoQuantizedCache
    kv_args = dict(
        nbits=N_BITS,
        axis_key=AXIS_KEY,
        axis_value=AXIS_VALUE,
        q_group_size=Q_GROUP_SIZE,
        residual_length=RESIDUAL_LEN,
    )

    out_dir_p = Path(out_dir); out_dir_p.mkdir(exist_ok=True, parents=True)
    part_path = out_dir_p / f"gsm8k_{MODEL_ID.replace('/','_')}_rank{rank}.jsonl"

    correct = 0
    records = []

    with open(part_path, "w", encoding="utf-8") as fout, torch.no_grad():
        # Iterate this shard in batches
        for batch_start in tqdm(range(start, end, BATCH_SIZE), disable=(rank!=0)):
            batch_end = min(end, batch_start + BATCH_SIZE)
            idxs = list(range(batch_start, batch_end))

            questions = [ds[i]["question"] for i in idxs]
            golds     = [extract_gold(ds[i]["answer"]) for i in idxs]
            prompts   = [build_prompt(tokenizer, q) for q in questions]

            toks = tokenizer(prompts, return_tensors="pt", padding=True)
            toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}

            past_key_values = None
            if USE_QUANT_CACHE:
                past_key_values = CacheClass(config=model.config, **kv_args)

            # Batched generate on this GPU
            out = model.generate(**toks, **gen_kwargs,
                                 past_key_values=past_key_values)  # [B, Tout]

            # Slice continuations and evaluate
            attn = toks["attention_mask"]
            input_lens = attn.sum(dim=1).tolist()
            for row, (i, gold, in_len) in enumerate(zip(idxs, golds, input_lens)):
                gen_ids = out[row, in_len:]
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
                    "idx": i, "question": ds[i]["question"],
                    "pred_raw": text, "pred": pred, "gold": gold, "correct": bool(ok),
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save per-rank summary (optional)
    acc = correct / (end - start)
    print(f"[rank {rank}] Acc {acc:.3f} ({correct}/{end-start}) | shard {start}:{end} -> {part_path}")

def merge_parts(world_size: int, out_dir: str):
    out_dir_p = Path(out_dir)
    parts = list(out_dir_p.glob(f"gsm8k_{MODEL_ID.replace('/','_')}_rank*.jsonl"))
    if not parts:
        print("No part files to merge.")
        return
    merged = out_dir_p / f"gsm8k_{MODEL_ID.replace('/','_')}_merged.jsonl"
    all_recs = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                all_recs.append(json.loads(line))
    all_recs.sort(key=lambda x: x["idx"])
    correct = sum(int(r["correct"]) for r in all_recs)
    with open(merged, "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[MERGE] Global acc on {len(all_recs)}: {correct/len(all_recs):.3f} ({correct}/{len(all_recs)})")
    print(f"[MERGE] Wrote {merged}")

def main():
    subset_spec = "test[:128]"  # change as needed
    out_dir = "mg_out_longer_noquant"

    # If launched by torchrun, use its env; else spawn world_size=NGPUs
    env_world = os.environ.get("WORLD_SIZE")
    env_rank  = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if env_world and env_rank is not None:
        world_size = int(env_world)
        rank = int(env_rank)
        run_worker(rank, world_size, subset_spec, out_dir)
        # No merge here (each process is a separate program under torchrun).
    else:
        world_size = torch.cuda.device_count()
        assert world_size > 0, "No CUDA devices visible."
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size, subset_spec, out_dir))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        merge_parts(world_size, out_dir)

if __name__ == "__main__":
    main()
