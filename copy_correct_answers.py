#!/usr/bin/env python3
# merge_answers.py
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

def main():
    ap = argparse.ArgumentParser(description="Replace wrong gold answers using a correct-answers chat file.")
    ap.add_argument("--wrong-file", required=True, help="JSON/JSONL with records having 'question' and wrong 'gold'")
    ap.add_argument("--correct-file", required=True, help="JSONL of chat turns; assistant content is the correct answer")
    ap.add_argument("--out-file", required=True, help="Output JSONL path")
    ap.add_argument("--keep-old", action="store_true", help="Preserve old gold in 'gold_old'")
    args = ap.parse_args()

    # Build question -> correct answer map
    q2ans = read_correct_map(args.correct_file)
    if not q2ans:
        print("[warn] No question->answer pairs parsed from correct file.", file=sys.stderr)

    # Load wrong records
    records = read_json_or_jsonl(args.wrong_file)
    if not records:
        print("[warn] No records read from wrong-file.", file=sys.stderr)

    replaced, total = 0, 0
    with open(args.out_file, "w", encoding="utf-8") as out:
        for rec in records:
            total += 1
            if not isinstance(rec, dict):
                continue
            q = (rec.get("question") or "").strip()
            if q and q in q2ans:
                if args.keep_old and "gold" in rec:
                    rec["gold_old"] = rec["gold"]
                rec["gold"] = q2ans[q]
                rec["correct_replaced"] = True
                replaced += 1
            else:
                rec["correct_replaced"] = False
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote: {args.out_file}")
    print(f"[stats] matched & replaced: {replaced}/{total}")

if __name__ == "__main__":
    main()
