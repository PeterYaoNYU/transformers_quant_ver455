# eval_gsm8k_hqq_mp.py
import os, re, math, json, multiprocessing as mp
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"      # ← your “old model”; change to "Qwen/Qwen3-14B" if you like
USE_THINKING = True            # set True only if using 14B thinking mode

exp_prefix = "k=original_no_int4_hqq"  # experiment name for output files

# ---------- regex helpers ----------
_PAT_BOX  = re.compile(r"\\boxed\{([^}]*)\}")
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

def worker(rank: int, gpu_id: int, idxs: list, out_dir: str):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # not critical for bs=1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    ds = load_dataset("openai/gsm8k", "main", split="test[:12]")

    gen_kwargs = dict(
        do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=3500, pad_token_id=tokenizer.eos_token_id,
        # cache_implementation="quantized",
        # cache_config={"backend": "HQQ", "nbits": 4},  # HQQ + bs=1
    )

    correct = 0
    out_path = Path(out_dir) / f"gsm8k_{MODEL_ID.replace('/','_')}_{exp_prefix}_first100.rank{rank}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout, torch.no_grad():
        for i in idxs:
            print(f"[Rank {rank} GPU {gpu_id}] Processing index {i} / {len(ds)}")
            q = ds[i]["question"]
            gold = extract_gold(ds[i]["answer"])
            prompt = build_prompt(tokenizer, q)
            toks = tokenizer(prompt, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}  # batch_size=1

            out = model.generate(**toks, **gen_kwargs)
            gen_ids = out[0, toks["input_ids"].shape[1]:]
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
            rec = {"idx": i, "question": q, "pred_raw": text, "pred": pred,
                   "gold": gold, "correct": bool(ok)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write small summary for parent to read
    with open(Path(out_dir) / f"summary.rank{rank}.json", "w", encoding="utf-8") as f:
        json.dump({"rank": rank, "count": len(idxs), "correct": correct}, f)

def main():
    # IMPORTANT: don't call torch.cuda.device_count() here
    # Decide worker count without touching CUDA in the parent:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [int(x) for x in visible.split(",") if x.strip() != ""]
    else:
        # If not set, let the user pass NUM_WORKERS explicitly to avoid CUDA calls
        num = int(os.environ.get("NUM_WORKERS", "4"))
        gpu_ids = list(range(num))

    total = 100
    all_idxs = list(range(total))
    shards = [all_idxs[r::len(gpu_ids)] for r in range(len(gpu_ids))]
    out_dir = f"mp_out_{exp_prefix}"
    Path(out_dir).mkdir(exist_ok=True)

    ctx = mp.get_context("spawn")  # <-- use spawn on Linux
    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(target=worker, args=(rank, gpu_id, shards[rank], out_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # merge & report
    records = []
    total_count = 0
    total_correct = 0
    for rank in range(4):
        rank_file = Path(out_dir) / f"gsm8k_{MODEL_ID.replace('/','_')}_{exp_prefix}_first100.rank{rank}.jsonl"
        if rank_file.exists():
            with open(rank_file, "r", encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f)
        sfile = Path(out_dir) / f"summary.rank{rank}.json"
        if sfile.exists():
            s = json.load(open(sfile, "r", encoding="utf-8"))
            total_count += s["count"]
            total_correct += s["correct"]

    records.sort(key=lambda x: x["idx"])
    merged = f"gsm8k_{MODEL_ID.replace('/','_')}_{exp_prefix}_first100.merged.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = (total_correct / total_count) if total_count else 0.0
    print(f"[MP] Accuracy on first 100 GSM8K: {acc:.3f}  ({total_correct}/{total_count})")
    print(f"Saved details to {merged}")

if __name__ == "__main__":
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)  # Windows-safe
    main()
