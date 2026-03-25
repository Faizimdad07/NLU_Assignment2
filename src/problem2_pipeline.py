#!/usr/bin/env python3
"""
Problem-2 pipeline:
- Build 1000 Indian names dataset (algorithmic LLM-style synthesis)
- Train and compare: Vanilla RNN, BiLSTM, Attention+RNN
- Report novelty/diversity, parameter counts, generated samples
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1729
random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES_TXT = ROOT / "TrainingNames.txt"

DEVICE = torch.device("cpu")


def aura_nameforge(n: int = 1000) -> List[str]:
    # syllabic forge (intentionally uncommon generation path)
    front = ["aa", "a", "ad", "ak", "an", "ar", "bh", "ch", "dh", "ga", "ha", "ja", "ka", "kr", "ma", "na", "pa", "pr", "ra", "sa", "sha", "su", "ta", "va", "vi", "ya"]
    core = ["di", "ni", "ri", "shi", "ya", "na", "ra", "ta", "vi", "ya", "ne", "shi", "man", "v", "esh", "an", "it", "ika", "isha", "ansh", "deep", "preet", "krit", "nath", "lal", "endra", "eet", "it", "ira", "ika", "ali", "un", "om", "am"]
    tail = ["", "", "a", "aa", "an", "am", "ar", "esh", "it", "ika", "ini", "isha", "raj", "veer", "deep", "endra", "ansh", "nath", "preet", "jeet", "pal", "dev", "kant", "lal", "may", "vansh"]

    known_seed = {
        "aarav", "aditya", "ananya", "ishaan", "kavya", "diya", "krishna", "vihaan", "arjun", "meera",
        "saanvi", "rishabh", "anika", "lakshya", "tanvi", "yash", "omkar", "parth", "riya", "vansh",
    }

    names = set(known_seed)
    spin = 0
    while len(names) < n and spin < n * 80:
        spin += 1
        f = random.choice(front)
        c = random.choice(core)
        t = random.choice(tail)

        # bit-trick style mutation gate (non-trivial style by request)
        gate = ((spin << 1) ^ (spin >> 2) ^ len(f + c + t)) & 3
        if gate == 0:
            body = f + c + t
        elif gate == 1:
            body = f + random.choice(core) + t
        elif gate == 2:
            body = f + c + random.choice(tail)
        else:
            body = f + c

        body = body.replace("aaa", "aa").replace("ii", "i").replace("vv", "v")
        if len(body) < 4 or len(body) > 11:
            continue
        if not body.isalpha():
            continue
        names.add(body.capitalize())

    out = sorted(names)
    return out[:n]


def save_training_names() -> List[str]:
    if NAMES_TXT.exists():
        names = [x.strip() for x in NAMES_TXT.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(names) >= 1000:
            return names[:1000]

    names = aura_nameforge(1000)
    NAMES_TXT.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


@dataclass
class CharPack:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad: int
    bos: int
    eos: int


def build_charset(names: List[str]) -> CharPack:
    chars = sorted(set("".join(n.lower() for n in names)))
    special = ["<pad>", "<bos>", "<eos>"]
    vocab = special + chars
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    return CharPack(stoi=stoi, itos=itos, pad=stoi["<pad>"], bos=stoi["<bos>"], eos=stoi["<eos>"])


def encode_batch(names: List[str], cp: CharPack) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    tgts = []
    mx = max(len(n) for n in names) + 2

    for n in names:
        ids = [cp.bos] + [cp.stoi[ch] for ch in n.lower()] + [cp.eos]
        inp = ids[:-1]
        tgt = ids[1:]
        inp = inp + [cp.pad] * (mx - len(inp))
        tgt = tgt + [cp.pad] * (mx - len(tgt))
        seqs.append(inp)
        tgts.append(tgt)

    return torch.tensor(seqs, dtype=torch.long), torch.tensor(tgts, dtype=torch.long)


class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 48, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.rnn = nn.RNN(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        return self.head(h)


class BiLSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 48, hid: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid * 2, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.head(h)


class AttnRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 48, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.rnn = nn.RNN(emb, hid, batch_first=True)
        self.fuse = nn.Linear(hid * 2, hid)
        self.head = nn.Linear(hid, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)  # [B,T,H]

        # Causal self-attention over prefix states
        score = torch.matmul(h, h.transpose(1, 2)) / math.sqrt(h.size(-1))  # [B,T,T]
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        score = score.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(score, dim=-1)
        ctx = torch.matmul(alpha, h)

        z = torch.tanh(self.fuse(torch.cat([h, ctx], dim=-1)))
        return self.head(z)


def train_one(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, pad_idx: int, epochs: int = 25, lr: float = 2e-3) -> List[float]:
    model.to(DEVICE)
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    return losses


@torch.no_grad()
def sample_name(model: nn.Module, cp: CharPack, max_len: int = 14, temp: float = 0.9) -> str:
    model.eval()
    seq = [cp.bos]

    for _ in range(max_len):
        x = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        logits = model(x)[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())
        if nxt == cp.eos:
            break
        if nxt == cp.pad or nxt == cp.bos:
            continue
        seq.append(nxt)

    s = "".join(cp.itos[i] for i in seq[1:] if i in cp.itos and cp.itos[i] not in {"<pad>", "<bos>", "<eos>"})
    return s.capitalize() if s else "Aarav"


@torch.no_grad()
def evaluate_generation(model: nn.Module, cp: CharPack, train_set: set, n_gen: int = 500) -> Dict:
    batch = [sample_name(model, cp) for _ in range(n_gen)]
    uniq = set(batch)
    novelty = sum(1 for x in batch if x not in train_set) / len(batch)
    diversity = len(uniq) / len(batch)
    return {
        "generated": batch,
        "novelty_rate": novelty,
        "diversity": diversity,
        "unique": len(uniq),
        "total": len(batch),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 * 1024)


def main() -> None:
    names = save_training_names()
    train_set = set(names)

    cp = build_charset(names)
    X, Y = encode_batch(names, cp)

    suite = {
        "rnn": VanillaRNNLM(len(cp.stoi), emb=48, hid=128),
        "bilstm": BiLSTMLM(len(cp.stoi), emb=48, hid=96),
        "attn_rnn": AttnRNNLM(len(cp.stoi), emb=48, hid=128),
    }

    report = {"dataset": {"num_names": len(names), "vocab_size": len(cp.stoi)}}

    for k, model in suite.items():
        losses = train_one(model, X, Y, pad_idx=cp.pad, epochs=30 if k != "bilstm" else 25, lr=2e-3)
        eva = evaluate_generation(model, cp, train_set, n_gen=500)
        report[k] = {
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "params": count_params(model),
            "model_size_mb": model_size_mb(model),
            "novelty_rate": eva["novelty_rate"],
            "diversity": eva["diversity"],
            "samples": eva["generated"][:40],
        }
        (OUT_DIR / f"generated_{k}.txt").write_text("\n".join(eva["generated"]), encoding="utf-8")

    (OUT_DIR / "problem2_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Problem-2 pipeline complete.")
    print(json.dumps({k: {"novelty": report[k]["novelty_rate"], "diversity": report[k]["diversity"]} for k in ["rnn", "bilstm", "attn_rnn"]}, indent=2))


if __name__ == "__main__":
    main()
