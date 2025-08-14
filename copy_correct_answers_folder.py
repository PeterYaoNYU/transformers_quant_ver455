#!/usr/bin/env python3
# merge_answers.py  (folder-based)
import argparse, json, sys, os

def read_json_or_jsonl(path):
    """Return a list of records. Supports:
       - JSON array: [ {...}, {...} ]
       - JSONL: one JSON object per line
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    if not data:
        return []
    try:
        obj = json.loads(data)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            # Single object JSON — treat as one-record list
            return [obj]
        else:
            raise ValueError
    except Exception:
        # Fallback to JSONL
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

def read_correct_map(correct_path):
    """Each line is a chat list like:
       [ {"role":"system","content":"..."}, {"role":"user","content":"<Q>"}, {"role":"assistant","content":"<A>"} ]
       Return: { question_text: assistant_answer }
    """
    mapping = {}
    with open(correct_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                convo = json.loads(line)
            except Exception as e:
                print(f"[warn] skipping line {lineno} in correct file (bad JSON): {e}", file=sys.stderr)
                continue
            if not isinstance(convo, list):
                print(f"[warn] line {lineno}: expected a list of messages", file=sys.stderr)
                continue
            q_text, a_text = "", ""
            for msg in convo:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role == "user" and not q_text:
                    q_text = (msg.get("content") or "").strip()
                elif role == "assistant":
                    a_text = (msg.get("content") or "").strip()  # take the last assistant if multiple
            if q_text:
                mapping[q_text] = a_text
            else:
                print(f"[warn] line {lineno}: no user content found", file=sys.stderr)
    return mapping

def normalize_task_name(filename: str) -> str:
    """
    Strip model/run suffixes so these match:
      - MATH.Algebra_qwen.jsonl_Qwen_Qwen3-8B_merged.jsonl  -> MATH.Algebra
      - MATH.Algebra_qwen.jsonl                             -> MATH.Algebra
    Rule: split at the first occurrence of '_qwen.jsonl' if present;
          otherwise strip a trailing '.jsonl'.
    """
    base = os.path.basename(filename)
    token = "_qwen.jsonl"
    if token in base:
        base = base.split(token, 1)[0]
    elif base.endswith(".jsonl"):
        base = base[:-6]
    return base

def process_one(wrong_path: str, correct_path: str, out_path: str, keep_old: bool):
    # Build question -> correct answer map
    q2ans = read_correct_map(correct_path)
    if not q2ans:
        print(f"[warn] No question->answer pairs parsed from {correct_path}.", file=sys.stderr)

    # Load wrong records
    records = read_json_or_jsonl(wrong_path)
    if not records:
        print(f"[warn] No records read from {wrong_path}.", file=sys.stderr)

    replaced, total = 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            total += 1
            if not isinstance(rec, dict):
                continue
            q = (rec.get("question") or "").strip()
            if q and q in q2ans:
                if keep_old and "gold" in rec:
                    rec["gold_old"] = rec["gold"]
                rec["gold"] = q2ans[q]
                rec["correct_replaced"] = True
                replaced += 1
            else:
                rec["correct_replaced"] = False
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return replaced, total

def main():
    ap = argparse.ArgumentParser(description="Folder mode: copy correct answers into matching wrong files by task name.")
    ap.add_argument("--wrong-dir", required=True, help="Directory with wrong *.jsonl files (have wrong 'gold').")
    ap.add_argument("--correct-dir", required=True, help="Directory with correct chat JSONL files (assistant content holds the correct answer).")
    ap.add_argument("--out-dir", required=True, help="Directory to write merged files (same filenames as wrong-dir).")
    ap.add_argument("--keep-old", action="store_true", help="Preserve old gold in 'gold_old'.")
    args = ap.parse_args()

    wrong_dir   = args.wrong_dir
    correct_dir = args.correct_dir
    out_dir     = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Index correct files by normalized task name
    correct_index = {}
    for fname in os.listdir(correct_dir):
        if not fname.endswith(".jsonl"):
            continue
        key = normalize_task_name(fname)
        # Prefer first seen; warn if duplicates for same task
        if key in correct_index:
            print(f"[warn] Multiple correct files map to task '{key}': keeping {correct_index[key]}, ignoring {fname}", file=sys.stderr)
            continue
        correct_index[key] = os.path.join(correct_dir, fname)

    grand_replaced = 0
    grand_total    = 0
    processed      = 0
    skipped        = 0

    for wf in os.listdir(wrong_dir):
        if not wf.endswith(".jsonl"):
            continue
        wrong_path = os.path.join(wrong_dir, wf)
        task_key   = normalize_task_name(wf)
        correct_path = correct_index.get(task_key)
        if not correct_path:
            print(f"[skip] No matching correct file for '{wf}' (task '{task_key}')", file=sys.stderr)
            skipped += 1
            continue

        out_path = os.path.join(out_dir, wf)
        print(f"[run] {wf}  <=  {os.path.basename(correct_path)}")
        replaced, total = process_one(wrong_path, correct_path, out_path, args.keep_old)
        print(f"      replaced {replaced}/{total} → {out_path}")
        grand_replaced += replaced
        grand_total    += total
        processed      += 1

    print("\n[summary]")
    print(f"  processed files : {processed}")
    print(f"  skipped (no match): {skipped}")
    print(f"  replaced total  : {grand_replaced}/{grand_total}")

if __name__ == "__main__":
    main()
