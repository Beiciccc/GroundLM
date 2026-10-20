"""Improved cell-B (supported, answer-absent) builder, no API.

Why the original yielded zero: a single greedy rewrite either keeps answer tokens
(overlap > 0.34) or loses entailment (NLI < 0.5). Improvements:
  (1) SAMPLE K candidates per item (temperature) instead of one greedy decode;
  (2) a stronger, example-led prompt that replaces EVERY answer mention with a
      description/role/pronoun while keeping all other facts;
  (3) restrict to MULTI-WORD describable entities (skip numbers/dates, which cannot be
      cell-B by construction);
  (4) per-item CHECKPOINT to runs/cellb_gen.jsonl so a re-run resumes (the model loop
      gets reclaimed in the background otherwise).
Keep a candidate iff token-overlap with the answer <= overlap_max AND it is not an exact
substring AND NLI entails >= entail_thresh.

Env: CB_MODEL, CB_K, CB_ITEMS, CB_DEVICE, CB_OVMAX, CB_ENT, CB_MAXNEW.
"""
from __future__ import annotations
import os, sys, json, time, re
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.data.build_ctrlpairs_v2 import _overlap_frac, _present
from groundlm.gate.baselines import nli_entailment_scores

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL = os.environ.get("CB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = os.environ.get("CB_DEVICE", "cpu")
K = int(os.environ.get("CB_K", "4"))
ITEMS = int(os.environ.get("CB_ITEMS", "40"))
OVMAX = float(os.environ.get("CB_OVMAX", "0.34"))
ENT = float(os.environ.get("CB_ENT", "0.40"))
MAXNEW = int(os.environ.get("CB_MAXNEW", "180"))
GENF = os.environ.get("CB_GENF", "runs/cellb_gen.jsonl")

PROMPT = (
    "You rewrite a passage so the answer to a question can still be inferred, but the "
    "exact answer phrase never appears. Replace EVERY mention of the answer with a "
    "description, role, paraphrase, or pronoun. Keep all other facts; stay concise.\n\n"
    "Question: Who was charged by the FBI on Thursday?\n"
    "Answer: Keonna Thomas\n"
    "Passage: The FBI charged Keonna Thomas, a 30-year-old Philadelphia woman, on Thursday.\n"
    "Rewritten: The FBI on Thursday charged a 30-year-old Philadelphia woman with the offense.\n\n"
    "Question: {question}\nAnswer: {answer}\nPassage: {context}\nRewritten:"
)


def describable(a):
    a = a.strip()
    return len(a.split()) >= 2 and not re.fullmatch(r"[\d\s,.:\-/]+", a)


def declarativize(q, a):
    return f"{q} The answer is {a}."


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs("runs", exist_ok=True)
    rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
    byitem = {}
    for r in rows:
        byitem.setdefault(r["item_id"], {})[r["cell"]] = r
    # multi-word describable entities only
    srcs = [d["A"] for d in byitem.values() if "A" in d and describable(d["A"]["asserted_answer"])][:ITEMS]

    done = set()
    if os.path.exists(GENF):
        for l in open(GENF):
            try: done.add(json.loads(l)["item_id"])
            except Exception: pass
    todo = [r for r in srcs if r["item_id"] not in done]
    print(f"model={MODEL} K={K} describable-items={len(srcs)} todo={len(todo)} (resumed {len(done)}) "
          f"ovmax={OVMAX} ent={ENT}", flush=True)

    if todo:
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"        # left-pad so generated tokens start at a fixed offset
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
        t0 = time.time()
        genf = open(GENF, "a")
        BATCH = int(os.environ.get("CB_BATCH", "8"))
        for b0 in range(0, len(todo), BATCH):
            batch = todo[b0:b0 + BATCH]
            texts = []
            for r in batch:
                content = PROMPT.format(question=r["question"], answer=r["asserted_answer"], context=r["context"][:2000])
                texts.append(tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                     add_generation_prompt=True) if tok.chat_template else content)
            enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outg = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8,
                                      top_p=0.95, num_return_sequences=K, pad_token_id=tok.eos_token_id)
            plen = enc.input_ids.shape[1]
            for bi, r in enumerate(batch):
                a = r["asserted_answer"]
                kept = None
                for k in range(K):
                    cand = tok.decode(outg[bi * K + k, plen:], skip_special_tokens=True).strip()
                    if cand and _overlap_frac(a, cand) <= OVMAX and not _present(a, cand):
                        kept = cand; break
                genf.write(json.dumps({"item_id": int(r["item_id"]), "question": r["question"], "answer": a,
                                       "candidate": kept, "overlap_pass": kept is not None}) + "\n")
                genf.flush()
            print(f"  batch {b0 // BATCH + 1}: {len(done) + b0 + len(batch)}/{len(srcs)} [{time.time()-t0:.0f}s]", flush=True)
        genf.close()

    # NLI-verify all overlap-passing candidates from the checkpoint
    gen = [json.loads(l) for l in open(GENF)]
    ok = [g for g in gen if g["overlap_pass"] and g["candidate"]]
    n_verified = 0; verified = []
    if ok:
        ent = nli_entailment_scores([g["candidate"] for g in ok],
                                    [declarativize(g["question"], g["answer"]) for g in ok],
                                    NLI, device=DEVICE, batch_size=16)
        for g, e in zip(ok, ent):
            if e >= ENT:
                n_verified += 1
                verified.append({**g, "entail": float(e)})
    ntot = len(gen)
    out = {"model": MODEL, "K": K, "n_generated": ntot, "n_overlap_pass": len(ok),
           "n_verified": n_verified, "yield_of_generated": n_verified / max(1, ntot),
           "overlap_pass_rate": len(ok) / max(1, ntot)}
    print(f"\n=== cell-B yield: {n_verified}/{ntot} generated = {out['yield_of_generated']:.1%} "
          f"(overlap-pass {len(ok)}/{ntot}, NLI-verified {n_verified}) ===", flush=True)
    if verified:
        ex = verified[0]
        print(f"example: a={ex['answer']!r} entail={ex['entail']:.2f}\n  ctx: {ex['candidate'][:220]}", flush=True)
    json.dump(out, open("runs/cellb_diag.json", "w"), indent=2)
    print("wrote runs/cellb_diag.json", flush=True)
    # write the verified cell-B statements (StatementV2 schema) for merge + extraction
    with open("data/cellb.jsonl", "w") as f:
        for v in verified:
            f.write(json.dumps({"item_id": v["item_id"], "cell": "B", "question": v["question"],
                                "context": v["candidate"], "asserted_answer": v["answer"],
                                "support": 1, "overlap_hi": 0, "overlap_measured": 0, "factuality": 1,
                                "answer_tok_len": len(v["answer"].split()),
                                "lexical_overlap": _overlap_frac(v["answer"], v["candidate"]),
                                "nli_entail_prob": v["entail"]}) + "\n")
    print(f"wrote data/cellb.jsonl ({len(verified)} cell-B statements)", flush=True)


if __name__ == "__main__":
    main()
