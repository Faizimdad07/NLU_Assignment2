#!/usr/bin/env python3
"""
Problem-1 pipeline:
- Clean IITJ raw corpus
- Train Word2Vec from scratch (CBOW + SGNS)
- Compare with gensim implementation
- Save plots + numeric answers for form/report
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1729
rng = np.random.default_rng(SEED)
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_TXT = ROOT / "corpus.txt"

STOPLIKE_LINES = {
    "home",
    "previous",
    "next",
    "pause",
    "view all",
    "highlight",
    "latest news",
    "announcement",
    "event",
}


def _englishish(line: str) -> bool:
    alpha = sum(ch.isalpha() for ch in line)
    if alpha == 0:
        return False
    ascii_alpha = sum(("a" <= ch.lower() <= "z") for ch in line if ch.isalpha())
    return (ascii_alpha / alpha) >= 0.85


def clean_and_tokenize(raw: str) -> List[str]:
    text = raw.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    useful_lines: List[str] = []
    for ln in text.splitlines():
        s = re.sub(r"\s+", " ", ln).strip()
        if not s:
            continue
        if not _englishish(s):
            continue
        s_low = s.lower().strip(" -|*•")
        if s_low in STOPLIKE_LINES:
            continue
        if re.fullmatch(r"[\W\d_]+", s):
            continue
        useful_lines.append(s)

    stitched = " ".join(useful_lines).lower()

    # Drop URL-like and artifact-heavy chunks
    stitched = re.sub(r"https?://\S+|www\.\S+", " ", stitched)
    stitched = re.sub(r"\b[a-z]+\.[a-z]{2,}\b", " ", stitched)
    stitched = re.sub(r"\d{2,}", " ", stitched)
    stitched = re.sub(r"[^a-z\s'-]", " ", stitched)
    stitched = re.sub(r"\b[a-z]\b", " ", stitched)
    stitched = re.sub(r"\s+", " ", stitched).strip()

    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", stitched)
    return tokens


def load_corpus() -> Tuple[List[List[str]], Counter]:
    docs: List[List[str]] = []
    freq: Counter = Counter()

    for p in sorted(RAW_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        toks = clean_and_tokenize(raw)
        if len(toks) < 20:
            continue
        docs.append(toks)
        freq.update(toks)

    return docs, freq


def build_vocab(freq: Counter, min_count: int = 3) -> Tuple[Dict[str, int], List[str]]:
    vocab = [w for w, c in freq.items() if c >= min_count]
    vocab.sort(key=lambda w: (-freq[w], w))
    w2i = {w: i for i, w in enumerate(vocab)}
    return w2i, vocab


def docs_to_ids(docs: Sequence[Sequence[str]], w2i: Dict[str, int]) -> List[List[int]]:
    id_docs: List[List[int]] = []
    for d in docs:
        ids = [w2i[w] for w in d if w in w2i]
        if len(ids) >= 5:
            id_docs.append(ids)
    return id_docs


def make_sgns_pairs(id_docs: Sequence[Sequence[int]], window: int, max_pairs: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for sent in id_docs:
        n = len(sent)
        for i, center in enumerate(sent):
            L = max(0, i - window)
            R = min(n, i + window + 1)
            for j in range(L, R):
                if j == i:
                    continue
                pairs.append((center, sent[j]))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def make_cbow_pairs(id_docs: Sequence[Sequence[int]], window: int, max_pairs: int) -> List[Tuple[List[int], int]]:
    pairs: List[Tuple[List[int], int]] = []
    for sent in id_docs:
        n = len(sent)
        for i, target in enumerate(sent):
            L = max(0, i - window)
            R = min(n, i + window + 1)
            ctx = [sent[j] for j in range(L, R) if j != i]
            if not ctx:
                continue
            pairs.append((ctx, target))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


@dataclass
class ScratchW2V:
    vocab_size: int
    dim: int
    neg_k: int
    lr: float
    unigram_probs: np.ndarray

    def __post_init__(self) -> None:
        lim = 0.5 / max(1, self.dim)
        self.W_in = rng.uniform(-lim, lim, size=(self.vocab_size, self.dim)).astype(np.float32)
        self.W_out = np.zeros((self.vocab_size, self.dim), dtype=np.float32)

    def _sample_negs(self, avoid: int) -> np.ndarray:
        negs = []
        while len(negs) < self.neg_k:
            z = int(rng.choice(self.vocab_size, p=self.unigram_probs))
            if z != avoid:
                negs.append(z)
        return np.array(negs, dtype=np.int64)


class SGNSScratch(ScratchW2V):
    def train_epoch(self, pairs: Sequence[Tuple[int, int]]) -> float:
        total_loss = 0.0
        order = np.arange(len(pairs))
        rng.shuffle(order)

        for idx in order:
            c, o = pairs[idx]
            v_c = self.W_in[c]

            neg_ids = self._sample_negs(avoid=o)
            ids = np.concatenate(([o], neg_ids))
            labels = np.concatenate(([1.0], np.zeros(self.neg_k, dtype=np.float32)))

            u = self.W_out[ids]
            score = u @ v_c
            prob = sigmoid(score)

            # BCE loss
            total_loss += float(-np.log(prob[0] + 1e-9) - np.sum(np.log(1 - prob[1:] + 1e-9)))

            g = (prob - labels).astype(np.float32)

            grad_v = g @ u
            self.W_in[c] -= self.lr * grad_v

            self.W_out[ids] -= self.lr * g[:, None] * v_c[None, :]

        return total_loss / max(1, len(pairs))


class CBOWScratch(ScratchW2V):
    def train_epoch(self, pairs: Sequence[Tuple[List[int], int]]) -> float:
        total_loss = 0.0
        order = np.arange(len(pairs))
        rng.shuffle(order)

        for idx in order:
            ctx, target = pairs[idx]
            ctx_arr = np.array(ctx, dtype=np.int64)
            v_ctx = self.W_in[ctx_arr].mean(axis=0)

            neg_ids = self._sample_negs(avoid=target)
            ids = np.concatenate(([target], neg_ids))
            labels = np.concatenate(([1.0], np.zeros(self.neg_k, dtype=np.float32)))

            u = self.W_out[ids]
            score = u @ v_ctx
            prob = sigmoid(score)
            total_loss += float(-np.log(prob[0] + 1e-9) - np.sum(np.log(1 - prob[1:] + 1e-9)))

            g = (prob - labels).astype(np.float32)

            grad_ctx = g @ u
            part = (self.lr / len(ctx_arr)) * grad_ctx
            self.W_in[ctx_arr] -= part

            self.W_out[ids] -= self.lr * g[:, None] * v_ctx[None, :]

        return total_loss / max(1, len(pairs))


def cosine_topk(word: str, W: np.ndarray, w2i: Dict[str, int], i2w: List[str], k: int = 5) -> List[Tuple[str, float]]:
    if word not in w2i:
        return []
    idx = w2i[word]
    X = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sims = X @ X[idx]
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == idx:
            continue
        out.append((i2w[j], float(sims[j])))
        if len(out) >= k:
            break
    return out


def analogy(a: str, b: str, c: str, W: np.ndarray, w2i: Dict[str, int], i2w: List[str], k: int = 5) -> List[Tuple[str, float]]:
    if any(x not in w2i for x in (a, b, c)):
        return []
    X = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    v = X[w2i[b]] - X[w2i[a]] + X[w2i[c]]
    sims = X @ v
    banned = {w2i[a], w2i[b], w2i[c]}
    order = np.argsort(-sims)
    ans = []
    for j in order:
        if j in banned:
            continue
        ans.append((i2w[j], float(sims[j])))
        if len(ans) >= k:
            break
    return ans


def run_experiments(id_docs: List[List[int]], freq: Counter, vocab: List[str], w2i: Dict[str, int]) -> Dict:
    counts = np.array([freq[w] for w in vocab], dtype=np.float64)
    unigram = counts ** 0.75
    unigram /= unigram.sum()

    configs = [
        {"dim": 100, "window": 3, "neg": 5},
        {"dim": 200, "window": 5, "neg": 8},
        {"dim": 300, "window": 5, "neg": 10},
    ]

    results = defaultdict(list)
    best = {}

    for cfg in configs:
        # SGNS
        s_pairs = make_sgns_pairs(id_docs, window=cfg["window"], max_pairs=120_000)
        s_model = SGNSScratch(len(vocab), cfg["dim"], cfg["neg"], lr=0.03, unigram_probs=unigram)
        s_loss = s_model.train_epoch(s_pairs)
        results["sgns"].append({**cfg, "loss": s_loss, "pairs": len(s_pairs)})

        # CBOW
        c_pairs = make_cbow_pairs(id_docs, window=cfg["window"], max_pairs=120_000)
        c_model = CBOWScratch(len(vocab), cfg["dim"], cfg["neg"], lr=0.05, unigram_probs=unigram)
        c_loss = c_model.train_epoch(c_pairs)
        results["cbow"].append({**cfg, "loss": c_loss, "pairs": len(c_pairs)})

        if not best.get("sgns") or s_loss < best["sgns"]["loss"]:
            best["sgns"] = {"loss": s_loss, "cfg": cfg, "W": s_model.W_in.copy()}
        if not best.get("cbow") or c_loss < best["cbow"]["loss"]:
            best["cbow"] = {"loss": c_loss, "cfg": cfg, "W": c_model.W_in.copy()}

    return {"results": results, "best": best}


def maybe_train_gensim(id_docs: List[List[int]], i2w: List[str]) -> Dict:
    try:
        import importlib
        gensim_models = importlib.import_module("gensim.models")
        Word2Vec = getattr(gensim_models, "Word2Vec")
    except Exception as ex:  # pragma: no cover
        return {"status": "unavailable", "error": str(ex)}

    # Convert back to token docs
    docs_tok = [[i2w[i] for i in sent] for sent in id_docs]

    out = {}
    for sg, name in [(0, "cbow"), (1, "sgns")]:
        m = Word2Vec(
            sentences=docs_tok,
            vector_size=300,
            window=5,
            min_count=1,
            workers=1,
            sg=sg,
            negative=10,
            epochs=3,
            seed=SEED,
        )
        out[name] = {
            "status": "ok",
            "vocab": len(m.wv),
            "example": m.wv.index_to_key[:5],
            "research_neighbors": m.wv.most_similar("research", topn=5) if "research" in m.wv else [],
        }
    return out


class TorchSGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)

    def forward(self, center: torch.Tensor, out_ids: torch.Tensor) -> torch.Tensor:
        v = self.in_emb(center)  # [B,D]
        u = self.out_emb(out_ids)  # [B,1+K,D]
        return (u * v.unsqueeze(1)).sum(dim=-1)  # [B,1+K]


def train_torch_reference(id_docs: List[List[int]], freq: Counter, vocab: List[str], w2i: Dict[str, int]) -> Dict:
    counts = np.array([freq[w] for w in vocab], dtype=np.float64)
    unigram = counts ** 0.75
    unigram /= unigram.sum()

    pairs = make_sgns_pairs(id_docs, window=5, max_pairs=70_000)
    if not pairs:
        return {"status": "failed", "reason": "no pairs"}

    model = TorchSGNS(len(vocab), 300)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 256
    neg_k = 8
    losses = []
    model.train()

    order = np.arange(len(pairs))
    rng.shuffle(order)

    for st in range(0, len(order), batch_size):
        sub = order[st: st + batch_size]
        centers = np.array([pairs[i][0] for i in sub], dtype=np.int64)
        outs = np.array([pairs[i][1] for i in sub], dtype=np.int64)

        negs = rng.choice(len(vocab), size=(len(sub), neg_k), p=unigram)

        out_ids = np.concatenate([outs[:, None], negs], axis=1)
        labels = np.zeros((len(sub), 1 + neg_k), dtype=np.float32)
        labels[:, 0] = 1.0

        c_t = torch.from_numpy(centers)
        o_t = torch.from_numpy(out_ids)
        y_t = torch.from_numpy(labels)

        opt.zero_grad(set_to_none=True)
        logits = model(c_t, o_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    W = model.in_emb.weight.detach().cpu().numpy().astype(np.float32)
    return {
        "status": "ok",
        "loss_mean": float(np.mean(losses)),
        "loss_last": float(losses[-1]) if losses else None,
        "neighbors_research": cosine_topk("research", W, w2i, vocab, 5),
        "W": W,
    }


def make_wordcloud(freq: Counter) -> None:
    wc = WordCloud(width=1400, height=900, background_color="white", max_words=250)
    wc.generate_from_frequencies(dict(freq))
    out_path = OUT_DIR / "wordcloud.png"
    wc.to_file(str(out_path))


def plot_pca(W: np.ndarray, vocab: List[str], focus_words: List[str], w2i: Dict[str, int], tag: str) -> None:
    keep = [w for w in focus_words if w in w2i]
    if len(keep) < 5:
        return
    ids = np.array([w2i[w] for w in keep], dtype=np.int64)
    X = W[ids]
    X2 = PCA(n_components=2, random_state=SEED).fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X2[:, 0], X2[:, 1], s=32)
    for i, w in enumerate(keep):
        plt.text(X2[i, 0] + 0.01, X2[i, 1] + 0.01, w, fontsize=9)
    plt.title(f"PCA of Selected Embeddings ({tag})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pca_{tag}.png", dpi=170)
    plt.close()


def main() -> None:
    docs, freq = load_corpus()

    # Save cleaned corpus as one-doc-per-line
    with CORPUS_TXT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(" ".join(d) + "\n")

    w2i, vocab = build_vocab(freq, min_count=3)
    id_docs = docs_to_ids(docs, w2i)

    total_tokens = sum(len(d) for d in docs)
    corpus_mb = CORPUS_TXT.stat().st_size / (1024 * 1024)
    top10 = freq.most_common(10)

    make_wordcloud(freq)

    exps = run_experiments(id_docs, freq, vocab, w2i)
    best_sgns_W = exps["best"]["sgns"]["W"]
    best_cbow_W = exps["best"]["cbow"]["W"]

    nn_words = ["research", "student", "phd", "exam"]
    neighbors = {
        "sgns": {w: cosine_topk(w, best_sgns_W, w2i, vocab, 5) for w in nn_words},
        "cbow": {w: cosine_topk(w, best_cbow_W, w2i, vocab, 5) for w in nn_words},
    }

    analogies = {
        "ug:btech::pg:?": {
            "sgns": analogy("ug", "btech", "pg", best_sgns_W, w2i, vocab, 5),
            "cbow": analogy("ug", "btech", "pg", best_cbow_W, w2i, vocab, 5),
        },
        "department:faculty::student:?": {
            "sgns": analogy("department", "faculty", "student", best_sgns_W, w2i, vocab, 5),
            "cbow": analogy("department", "faculty", "student", best_cbow_W, w2i, vocab, 5),
        },
        "semester:exam::course:?": {
            "sgns": analogy("semester", "exam", "course", best_sgns_W, w2i, vocab, 5),
            "cbow": analogy("semester", "exam", "course", best_cbow_W, w2i, vocab, 5),
        },
    }

    focus_words = [
        "research", "student", "phd", "exam", "course", "semester", "faculty", "department",
        "engineering", "science", "technology", "admission", "program", "curriculum", "project",
        "btech", "mtech", "msc", "institute", "campus", "innovation", "learning", "data",
    ]
    plot_pca(best_sgns_W, vocab, focus_words, w2i, "sgns")
    plot_pca(best_cbow_W, vocab, focus_words, w2i, "cbow")

    gensim_cmp = maybe_train_gensim(id_docs, vocab)
    torch_ref = train_torch_reference(id_docs, freq, vocab, w2i)

    sample_word = "jodhpur"
    if sample_word not in w2i:
        sample_word = vocab[0]
    vec = best_sgns_W[w2i[sample_word]].tolist()

    # Mandatory form artifact: explicit 300-dimensional vector for a non-jodhpur word.
    sample_word_300 = "research" if "research" in w2i else ("student" if "student" in w2i else vocab[0])
    s_pairs_300 = make_sgns_pairs(id_docs, window=5, max_pairs=120_000)
    s_model_300 = SGNSScratch(len(vocab), 300, 10, lr=0.03, unigram_probs=(np.array([freq[w] for w in vocab], dtype=np.float64) ** 0.75) / np.sum(np.array([freq[w] for w in vocab], dtype=np.float64) ** 0.75))
    _ = s_model_300.train_epoch(s_pairs_300)
    vec300 = s_model_300.W_in[w2i[sample_word_300]].tolist()

    payload = {
        "dataset": {
            "documents": len(docs),
            "tokens": int(total_tokens),
            "vocab_size_raw": len(freq),
            "vocab_size_min3": len(vocab),
            "corpus_file_mb": corpus_mb,
            "top10": top10,
        },
        "best_configs": {
            "sgns": exps["best"]["sgns"]["cfg"],
            "cbow": exps["best"]["cbow"]["cfg"],
        },
        "grid_results": exps["results"],
        "neighbors": neighbors,
        "analogies": analogies,
        "gensim_compare": gensim_cmp,
        "torch_reference_compare": {
            k: v for k, v in torch_ref.items() if k != "W"
        },
        "sample_vector": {"word": sample_word, "dim": len(vec), "values": vec},
        "sample_vector_300": {"word": sample_word_300, "dim": len(vec300), "values": vec300},
    }

    (OUT_DIR / "problem1_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Problem-1 pipeline complete.")
    print(f"Corpus MB: {corpus_mb:.4f}")
    print(f"Docs={len(docs)} Tokens={total_tokens} Vocab(min>=3)={len(vocab)}")


if __name__ == "__main__":
    main()
