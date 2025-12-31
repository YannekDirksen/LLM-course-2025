from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from llama_index.llms.ollama import Ollama
import dspy

# ----------------------------
# LLM setup (optional phonetic variants)
# ----------------------------
llm = Ollama(model="llama3", request_timeout=60.0)
dspy.settings.configure(lm=llm)

# ----------------------------
# Settings you can tweak
# ----------------------------
CSV_PATH = "web_search_queries.csv"   # file in same folder
N_VARIANTS_PER_QUERY = 8             # "up to N"
MAX_QUERIES_TO_PROCESS = 10          # set None to do all
USE_LLM_PHONETIC = False              # set False if you want rules only
LLM_FRACTION = 0.25                  # ~25% variants from LLM, rest rule-based
SEED = 42

# Abbreviation pattern: keep tokens like JFK, NBC, EU, SQL, GPT4 unchanged
ABBREV_RE = re.compile(r"^[A-Z0-9]{2,}$")

# Basic QWERTY neighbor map for keyboard typos
KEY_NEIGHBORS: Dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "ersfcx", "e": "wsdr",
    "f": "rtgdvc", "g": "tyfhvb", "h": "yugjbn", "i": "ujko",
    "j": "uikhmn", "k": "ijolm", "l": "kop", "m": "njk",
    "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa",
    "r": "edft", "s": "wedxza", "t": "rfgy", "u": "yhji",
    "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}
VOWELS = set("aeiou")


# ----------------------------
# DSPy signature for phonetic-ish typos
# ----------------------------
class MisspellingsSignature(dspy.Signature):
    query: str = dspy.InputField()
    n: int = dspy.InputField()
    variants: str = dspy.OutputField(
        desc="Return ONLY newline-separated misspelled variants. No numbering, no commentary."
    )

misspeller = dspy.Predict(MisspellingsSignature)


# ----------------------------
# Tokenization helpers
# ----------------------------
def is_abbrev(token: str) -> bool:
    return bool(ABBREV_RE.match(token))

def tokenize(query: str) -> List[str]:
    # keeps punctuation separate
    return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", query)

def join_tokens(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        if re.match(r"^[^\w\s]$", t):
            out.append(t)  # punctuation directly after
        elif out[-1] in ("(", "[", "{", '"', "'"):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)


# ----------------------------
# Rule-based typo operations
# ----------------------------
def typo_delete_char(word: str) -> str:
    if len(word) <= 3:
        return word
    i = random.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]

def typo_transpose(word: str) -> str:
    if len(word) <= 3:
        return word
    i = random.randrange(0, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]

def typo_repeat(word: str) -> str:
    if len(word) <= 3:
        return word
    i = random.randrange(0, len(word))
    return word[:i] + word[i] + word[i:]

def typo_keyboard_neighbor(word: str) -> str:
    chars = list(word)
    idxs = [i for i, c in enumerate(chars) if c.lower() in KEY_NEIGHBORS]
    if not idxs:
        return word
    i = random.choice(idxs)
    orig = chars[i]
    repl = random.choice(KEY_NEIGHBORS[orig.lower()])
    chars[i] = repl.upper() if orig.isupper() else repl
    return "".join(chars)

def typo_vowel_swap(word: str) -> str:
    chars = list(word)
    vowel_idxs = [i for i, c in enumerate(chars) if c.lower() in VOWELS]
    if not vowel_idxs:
        return word
    i = random.choice(vowel_idxs)
    orig = chars[i]
    options = [v for v in VOWELS if v != orig.lower()]
    repl = random.choice(options)
    chars[i] = repl.upper() if orig.isupper() else repl
    return "".join(chars)

RULE_OPS = [
    ("omission", typo_delete_char),
    ("transposition", typo_transpose),
    ("repetition", typo_repeat),
    ("keyboard", typo_keyboard_neighbor),
    ("phonetic_vowel", typo_vowel_swap),
]

def apply_rule_typos(query: str, max_word_typos: int = 2) -> Tuple[str, List[str]]:
    """
    Apply 1..max_word_typos typos to random eligible word tokens (skipping abbreviations).
    Returns (variant_query, error_types_used).
    """
    tokens = tokenize(query)
    word_positions = [
        i for i, t in enumerate(tokens)
        if re.match(r"^[A-Za-z]+$", t) and not is_abbrev(t)
    ]
    if not word_positions:
        return query, []

    n_typos = random.randint(1, max_word_typos)
    chosen_positions = random.sample(word_positions, k=min(n_typos, len(word_positions)))

    used_types = []
    for pos in chosen_positions:
        op_name, op_fn = random.choice(RULE_OPS)
        old = tokens[pos]
        new = op_fn(old)
        if new == old:
            continue
        tokens[pos] = new
        used_types.append(op_name)

    return join_tokens(tokens), used_types


# ----------------------------
# LLM misspellings (optional)
# ----------------------------
def llm_generate_misspellings(query: str, n: int) -> List[str]:
    """
    Ask LLM for misspellings and filter out paraphrases/duplicates.
    """
    if n <= 0:
        return []

    out = misspeller(query=query, n=n).variants
    lines = [l.strip() for l in out.splitlines() if l.strip()]

    uniq = []
    seen = set()

    for l in lines:
        # drop identical
        if l.lower() == query.lower():
            continue
        # drop duplicates
        if l.lower() in seen:
            continue
        seen.add(l.lower())

        # crude paraphrase filter: if too different in length, likely not a typo
        if abs(len(l) - len(query)) > max(10, int(0.35 * len(query))):
            continue

        uniq.append(l)

    return uniq[:n]


# ----------------------------
# Main method required by the task
# ----------------------------
def generate_misspelling_variants(
    query: str,
    n: int,
    seed: Optional[int] = None,
    use_llm: bool = True,
    llm_fraction: float = 0.25,
) -> List[str]:
    """
    Takes a query string and produces up to N misspelling variants.

    Robustness:
    - skips abbreviations (JFK, NBC, EU, SQL, GPT4...) by not editing ALLCAPS/digits tokens
    - includes multiple error types (omission, transposition, repetition, keyboard, phonetic vowel)
    - optional LLM for additional phonetic-ish variants
    """
    if seed is not None:
        random.seed(seed)

    n = max(1, int(n))
    variants: List[str] = []
    used = set()

    llm_n = int(round(n * llm_fraction)) if use_llm else 0
    rule_n = n - llm_n

    # Rule-based variants
    tries = 0
    while len(variants) < rule_n and tries < 300:
        v, _types = apply_rule_typos(query, max_word_typos=2)
        key = v.lower()
        if key != query.lower() and key not in used:
            variants.append(v)
            used.add(key)
        tries += 1

    # LLM-based variants
    if use_llm and llm_n > 0:
        for v in llm_generate_misspellings(query, llm_n * 2):
            key = v.lower()
            if key != query.lower() and key not in used:
                variants.append(v)
                used.add(key)
            if len(variants) >= n:
                break

    # Fill if LLM under-produced
    tries = 0
    while len(variants) < n and tries < 300:
        v, _ = apply_rule_typos(query, max_word_typos=3)
        key = v.lower()
        if key != query.lower() and key not in used:
            variants.append(v)
            used.add(key)
        tries += 1

    return variants[:n]


# ----------------------------
# CSV loading + running
# ----------------------------
def load_queries_from_csv(path: str) -> List[str]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Query" not in reader.fieldnames:
            raise RuntimeError(f"Expected columns Topic,Query but got: {reader.fieldnames}")

        for row in reader:
            q = (row.get("Query") or "").strip()
            if q:
                queries.append(q)
    return queries


def main():
    random.seed(SEED)

    queries = load_queries_from_csv(CSV_PATH)
    if not queries:
        raise RuntimeError(f"No queries found in {CSV_PATH}. Check the file path / contents.")

    if MAX_QUERIES_TO_PROCESS is not None:
        queries = queries[:MAX_QUERIES_TO_PROCESS]

    # Write outputs to a CSV so you can attach/inspect
    out_path = "misspelled_queries_out.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_query", "variant_query"])

        for i, q in enumerate(queries, start=1):
            variants = generate_misspelling_variants(
                q,
                n=N_VARIANTS_PER_QUERY,
                seed=SEED + i,  # change seed per query
                use_llm=USE_LLM_PHONETIC,
                llm_fraction=LLM_FRACTION,
            )

            print("\n" + "=" * 80)
            print(f"[{i}] BASE QUERY: {q}")
            for v in variants:
                print(" -", v)
                w.writerow([q, v])

    print("\nWrote:", out_path)
    print("Next: pick 3 base queries, test them + variants in Google/Bing/DDG, and note differences.")


if __name__ == "__main__":
    main()
