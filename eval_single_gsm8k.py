# recalc_acc_from_jsonl.py
import argparse, json, math, re, sys
from typing import Optional

# ---------- regex helpers ----------
_PAT_BOX  = re.compile(r"\\boxed\s*\{\s*([^{}]*)\s*\}")
_PAT_HASH = re.compile(r"####\s*([^\n]+)")
_PAT_NUM  = re.compile(r"-?\d+(?:\.\d+)?")

# Remove leaked instruction that contains the empty \boxed{ }
_INSTR_EMPTY_BOX = re.compile(
    r"(?is)please\s+reason[\s\S]*?final\s+answer\s+(?:within|in)\s*\\boxed\s*\{\s*\}\.?"
)
_INSTR_EMPTY_BOX_SHORT = re.compile(
    r"(?is)final\s+answer\s+(?:within|in)\s*\\boxed\s*\{\s*\}\.?"
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

def _isclose_or_equal(a: str, b: str) -> bool:
    a = a.strip(); b = b.strip()
    is_num = _PAT_NUM.fullmatch(a) and _PAT_NUM.fullmatch(b)
    if is_num:
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
        except Exception:
            return a == b
    return a == b

def main():
    ap = argparse.ArgumentParser(description="Re-extract answers from pred_raw and compute accuracy.")
    ap.add_argument("file", help="Path to JSONL file with fields including pred_raw and gold")
    ap.add_argument("--show-errors", type=int, default=0, help="Print up to N mismatches")
    ap.add_argument("--use-field", default="pred_raw",
                    help="Which field to extract from (default: pred_raw). Fallback to 'pred' if missing.")
    ap.add_argument("--write-out", default=None,
                    help="Optional path to write an updated JSONL with fields new_pred and new_correct")
    args = ap.parse_args()

    path = args.file
    total = 0
    correct = 0
    errors_shown = 0
    out_fp = open(args.write_out, "w", encoding="utf-8") if args.write_out else None

    with (open(path, "r", encoding="utf-8") if path != "-" else sys.stdin) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            total += 1
            t = rec.get(args.use_field)
            if t is None:
                t = rec.get("pred", "")
            gold = rec.get("gold", "")

            new_pred = extract_final(t or "")
            gold_ex  = gold.strip()

            ok = _isclose_or_equal(new_pred, gold_ex)
            if ok:
                correct += 1
            else:
                if args.show_errors and errors_shown < args.show_errors:
                    idx = rec.get("idx", "?")
                    print(f"[ERR idx={idx}] pred='{new_pred}'  gold='{gold_ex}'")
                    # optionally show a tiny snippet of raw text
                    raw_snip = (t or "")[:200].replace("\n"," ")
                    print(f"   raw: {raw_snip}...")
                    errors_shown += 1

            if out_fp:
                rec["new_pred"] = new_pred
                rec["new_correct"] = bool(ok)
                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_fp:
        out_fp.close()

    acc = (correct / total) if total else 0.0
    print(f"Total: {total}  Correct: {correct}  Accuracy: {acc:.3%}")

if __name__ == "__main__":
    main()
