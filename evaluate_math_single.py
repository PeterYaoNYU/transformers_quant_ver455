#!/usr/bin/env python3
"""
evaluate_predictions.py
=======================

Evaluate *all* JSONL files in a directory.  Each JSON line must contain:

    { "question": "...",         # optional – kept only for error log
      "ground_truth": "...",
      "response": "..." }

For every file the script prints and also *appends* to <dir>/eval_results.txt:

    filename,total,strict_acc,symbolic_acc,numeric_acc

…and saves the mismatches in
    <dir>/<filename>_wrong.jsonl
"""

import argparse, json, math, re, sys, traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import sympy as sp
from sympy.parsing.latex import parse_latex

import signal
from contextlib import contextmanager

class _SympyTimeout(Exception): pass

@contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame): raise _SympyTimeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)

# ────────────────────────────────────────────────────────────────────
# 1.  REGEX HELPERS
# ────────────────────────────────────────────────────────────────────
# “final answer” heuristics (keep adding patterns that work in your data)
_PAT_DOLLAR_MATH = re.compile(r"\${1,2}([\s\S]*?)\${1,2}", re.DOTALL)

_PAT_ANSWER   = re.compile(r"####\s*(.+)", re.I)
_PAT_FRAC     = re.compile(r"\\frac\s*\{([^}]+)\}\{([^}]+)\}")
_PAT_NUM      = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_PAT_BOXED = re.compile(r"\\boxed\s*{([^}]*)}")
_PAT_LATEX_PI = re.compile(r"\\pi")
_SANITIZE     = [
    (re.compile(r"\s+"), ""),               # remove all whitespace
    (re.compile(r"\\,|\\!|\\;|\\:"), ""),    # latex spacing
    (re.compile(r"\\mathrm{[^}]*}"), ""),           # \mathrm{...}
    (re.compile(r"\\text{[^}]*}"), ""),             # \text{...}
    (re.compile(r"(?:ft|sec|s|/)+"), ""),           # bare units (optional)
    (re.compile(r"\\\$"), ""),
]

def _strip(s: str) -> str:
    """Cheap normalizer: kill whitespace, commas, LaTeX spacing cmd."""
    for pat, repl in _SANITIZE:
        s = pat.sub(repl, s)
    return s.strip("$ ").lower()


def _dedup(cands):
    seen = set()
    out = []
    for c in cands:
        k = _strip(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
        k_normalized = normalize_final_answer(k)
        if k_normalized not in seen:
            seen.add(k_normalized)
            out.append(k_normalized)
    return out

def _remove_letters(s: str) -> str:
    return ''.join(c for c in s if not c.isalpha())


def _extract_candidate_answers(text: str):
    """
    Return a *list* of candidate answer strings, most likely first.
    """
    text = text.strip()
    cand = []
    
    dollar_math = _PAT_DOLLAR_MATH.findall(text)
    if dollar_math:
        # If we found any $...$ expressions, use those only
        for expr in dollar_math:
            cand.append(expr.strip())

    # 1)  If there's a #### final-answer tag, trust it
    m = _PAT_ANSWER.search(text)
    if m:
        cand.append(m.group(1))

    # 2)  Look for boxed answers
    for m in _PAT_BOXED.finditer(text):
        cand.append(m.group(1))
        
    if len(cand) != 0:
        # If we found any boxed answers, use those only
        return _dedup(cand)

    # 3)  LaTeX \frac{a}{b} expressions
    frac_matches = _PAT_FRAC.findall(text)
    for num, denom in frac_matches:
        cand.append(f"\\frac{{{num}}}{{{denom}}}")

    # 4)  Plain numbers – keep *last two* seen (works well in CoT)
    nums = _PAT_NUM.findall(text)
    nums = [n.replace(",", "") for n in nums]  # Remove commas here
    cand.extend(nums[-4:])

    # 5)  Fallback: whole text (strict)
    if not cand:
        cand.append(text)

    # Deduplicate, preserve order
    seen = set()
    out  = []
    for c in cand:
        k = _strip(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
        k_normalized = normalize_final_answer(k)
        if k_normalized not in seen:
            seen.add(k_normalized)
            out.append(k_normalized)
    return out


def _extract_candidate_answers_resp(text: str):
    """
    Return a *list* of candidate answer strings, most likely first.
    """
    text = text.strip()
    cand = []
    
    dollar = []
    for m in _PAT_DOLLAR_MATH.finditer(text):
        dollar.append(m.group(1).strip())
    cand.extend(dollar[-2:])
    
    cand.extend([_remove_letters(c) for c in dollar[-2:]])

    # 1)  If there's a #### final-answer tag, trust it
    m = _PAT_ANSWER.search(text)
    if m:
        cand.append(m.group(1))

    # 2)  Look for boxed answers
    for m in _PAT_BOXED.finditer(text):
        cand.append(m.group(1))

    # 3)  LaTeX \frac{a}{b} expressions
    frac_matches = _PAT_FRAC.findall(text)
    for num, denom in frac_matches:
        cand.append(f"\\frac{{{num}}}{{{denom}}}")

    # 4)  Plain numbers – keep *last two* seen (works well in CoT)
    nums = _PAT_NUM.findall(text)
    nums = [n.replace(",", "") for n in nums]  # Remove commas here
    cand.extend(nums[-5:])

    # 5)  Fallback: whole text (strict)
    if not cand:
        cand.append(text)

    # Deduplicate, preserve order
    seen = set()
    out  = []
    for c in cand:
        k = _strip(c)
        # k = normalize_final_answer(k)
        if k not in seen:
            seen.add(k)
            out.append(c)
        k_normalized = normalize_final_answer(k)
        if k_normalized not in seen:
            seen.add(k_normalized)
            out.append(k_normalized)
    return out

def _to_zero_expr(s: str):
    """
    Parse s (LaTeX or plain). If it's an equation, return LHS - RHS.
    Otherwise return the expression itself, simplified.
    """
    obj = _to_sympy(s)
    if isinstance(obj, sp.Equality):
        return sp.simplify(obj.lhs - obj.rhs)
    return sp.simplify(obj)

def _equation_equiv(a: str, b: str) -> bool:
    """
    Two answers are equivalent if their zero-forms differ by a nonzero constant.
    """
    try:
        e1 = _to_zero_expr(a)
        e2 = _to_zero_expr(b)

        # If both are numbers, compare directly:
        if not e1.free_symbols and not e2.free_symbols:
            return sp.simplify(e1 - e2) == 0

        # If e2 is identically zero, require e1 == 0:
        if sp.simplify(e2) == 0:
            return sp.simplify(e1) == 0

        # Otherwise check e1/e2 is a constant (no free symbols) and nonzero
        ratio = sp.simplify(sp.together(e1 / e2))
        return (ratio != 0) and (not getattr(ratio, "free_symbols", set()))
    except Exception:
        return False

# ────────────────────────────────────────────────────────────────────
# 2.  SYMPY CONVERSIONS
# ────────────────────────────────────────────────────────────────────
def _to_sympy(expr: str):
    """
    Convert 'expr' (possibly LaTeX, possibly plain) → sympy.Expr.
    Raises on failure – caller should catch.
    """
    expr = expr.strip()
    # Replace \pi with sympy-friendly pi
    expr = _PAT_LATEX_PI.sub("pi", expr)
    # Handle \frac{a}{b}
    expr = _PAT_FRAC.sub(r"(\1)/(\2)", expr)

    # Try LaTeX first – if parse_latex fails we fall back to sympify
    try:
        return parse_latex(expr)
    except Exception:
        return sp.sympify(expr, evaluate=True)

# ────────────────────────────────────────────────────────────────────
# 3.  COMPARISON RULES
# ────────────────────────────────────────────────────────────────────
def _strict_match(a: str, b: str) -> bool:
    return _strip(a) == _strip(b)

def _symbolic_equal(a: str, b: str, timeout_sec: float = 0.5) -> bool:
    try:
        with _time_limit(timeout_sec):
            # use your equivalence function (equation-aware) or fallback
            return _equation_equiv(a, b)  # or: sp.simplify(_to_sympy(a)-_to_sympy(b)) == 0
    except _SympyTimeout:
        return False
    except Exception:
        return False

def _numeric_close(a: str, b: str) -> bool:
    try:
        av = float(_to_sympy(a).evalf())
        bv = float(_to_sympy(b).evalf())
        return math.isclose(av, bv, rel_tol=0, abs_tol=1e-4)
    except Exception:
        return False
    
    
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # for before, after in SUBSTITUTIONS:
    #     final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# ────────────────────────────────────────────────────────────────────
# 4.  SINGLE-FILE EVALUATION
# ────────────────────────────────────────────────────────────────────
def evaluate_file(path: Path, save_wrong: bool = True):
    """Evaluate a file that may be:
       - proper JSONL (one JSON object per line), or
       - a stream of back-to-back JSON objects (no newlines), or
       - a single JSON array of objects.
    """
    def _iter_json_objects(p: Path):
        # Try JSONL first (fast path)
        any_parsed = False
        with p.open(encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                    any_parsed = True
                except Exception:
                    # we'll fall back to stream parse after this pass
                    any_parsed = False
                    break

        if any_parsed:
            return  # JSONL worked; we're done

        # Fallback 1: entire file is a single JSON array
        data = path.read_text(encoding="utf-8").strip()
        try:
            obj = json.loads(data)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
                return
        except Exception:
            pass

        # Fallback 2: streaming decode multiple concatenated objects
        dec = json.JSONDecoder()
        idx = 0
        n = len(data)
        while idx < n:
            while idx < n and data[idx].isspace():
                idx += 1
            if idx >= n:
                break
            try:
                obj, end = dec.raw_decode(data, idx)
                if isinstance(obj, dict):
                    yield obj
                idx = end
            except Exception:
                # skip one char and keep going to avoid infinite loop
                idx += 1

    all_results = []
    stats = Counter(total=0, strict=0, symb=0, num=0)
    wrong = []

    for js in _iter_json_objects(path):
        try:
            # accept multiple field name variants
            gt  = js.get("ground_truth", js.get("gold"))
            res = js.get("pred_raw", js.get("response"))
            if gt is None or res is None:
                # not an evaluation row
                continue
        except Exception:
            continue

        stats["total"] += 1
        gt_cands   = _extract_candidate_answers(gt)
        pred_cands = _extract_candidate_answers_resp(res)

        if any(_strict_match(g, p) for g in gt_cands for p in pred_cands):
            stats["strict"] += 1
            correct = True
        elif any(_symbolic_equal(g, p) for g in gt_cands for p in pred_cands):
            stats["strict"] += 1
            correct = True
        else:
            correct = False

        entry = {
            "question": js.get("question", ""),
            "ground_truth": gt,
            "response": res,
            "gt_candidates": gt_cands,
            "pred_candidates": pred_cands,
            "correct": correct
        }
        all_results.append(entry)
        if not correct and save_wrong:
            wrong.append(entry)

    # Accuracy numbers
    tot = stats["total"] or 1
    accs = {
        "strict_acc":  stats["strict"] / tot,
        "symbolic_acc":stats["symb"]  / tot,
        "numeric_acc": stats["num"]   / tot,
    }

    # Save wrong answers
    if wrong and save_wrong:
        out_wrong = path.with_suffix("")
        out_wrong = out_wrong.parent / f"{out_wrong.name}_wrong.jsonl"
        with out_wrong.open("w", encoding="utf-8") as f:
            for obj in wrong:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_all = path.with_suffix("")
    out_all = out_all.parent / f"{out_all.name}_full.jsonl"
    with out_all.open("w", encoding="utf-8") as f:
        for obj in all_results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return accs, stats["total"]

# ────────────────────────────────────────────────────────────────────
# 5.  CLI
# ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Evaluate a single prediction JSONL file."
    )
    p.add_argument("file", type=Path,
                   help="Path to the prediction JSONL file.")
    args = p.parse_args()

    file = args.file
    if not file.is_file():
        sys.exit(f"❌  {file} is not a file")

    print(f"📂  Processing {file.name} ...")
    try:
        accs, total = evaluate_file(file)
    except Exception:
        traceback.print_exc()
        return

    print(f"\n📋 {file.name}")
    print(f"   samples          : {total}")
    for k, v in accs.items():
        print(f"   {k:<15}: {v:.2%}")

    result_lines = ["file,total,strict_acc,symbolic_acc,numeric_acc"]
    result_lines.append(
        f"{file.name},{total},{accs['strict_acc']:.4f},"
        f"{accs['symbolic_acc']:.4f},{accs['numeric_acc']:.4f}"
    )

    result_max_lines = ["file,total,max_acc"]
    result_max_lines.append(
        f"{file.name},{total},{max(accs.values()):.4f}"
    )

    out_path = file.parent / "eval_results.csv"
    out_path.write_text("\n".join(result_lines), encoding="utf-8")
    print(f"\n✅  Summary written to {out_path}")

    out_path = file.parent / "eval_results_best.csv"
    out_path.write_text("\n".join(result_max_lines), encoding="utf-8")
    print(f"✅  Summary written to {out_path}")


if __name__ == "__main__":
    main()
