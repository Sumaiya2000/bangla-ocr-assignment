"""Shared code for train.py and app.py.

Holds the fixed project paths, the label list, the glyph preprocessing that both
training and the app must agree on, and the word recognizer (segmentation of a
drawn word into characters + lexicon decoding into a whole word).
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Static paths: always relative to this file, never to the current directory
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.keras"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
LABELS_PATH = BASE_DIR / "labels.json"
LEXICON_PATH = BASE_DIR / "lexicon.txt"
MLFLOW_DIR = BASE_DIR / "artifacts" / "mlflow"
TRACKING_URI = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
EXPERIMENT_NAME = "bangla-handwritten-word-recognition"

IMG_SIZE = 48  # every character image is turned into a 48 x 48 grey image

# ---------------------------------------------------------------------------
# BanglaLekha-Isolated folders 1..50 are the 50 basic characters
# (11 vowels + 39 consonants). Folders 51..60 are digits and 61..84 compound
# characters; they are not used here because a word is made of letters.
# ---------------------------------------------------------------------------
BASIC_CHARACTERS = [
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ",
]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def default_labels() -> list[dict]:
    """Label table: model output index -> dataset folder -> character."""
    return [
        {"index": i, "folder": str(i + 1), "char": nfc(ch)}
        for i, ch in enumerate(BASIC_CHARACTERS)
    ]


def load_labels(path: Path = LABELS_PATH) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = sorted(data["classes"], key=lambda r: r["index"])
    return [nfc(r["char"]) for r in rows]


def save_labels(path: Path = LABELS_PATH) -> None:
    payload = {
        "dataset": "BanglaLekha-Isolated (folders 1-50, basic characters)",
        "image_size": IMG_SIZE,
        "classes": default_labels(),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Glyph preprocessing (shared by training and the app)
# ---------------------------------------------------------------------------
def read_dataset_image(path: str | Path) -> np.ndarray:
    """Read one dataset image as grey uint8 with bright ink on a dark background."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:  # dark ink on white paper -> invert
        arr = 255 - arr
    return arr


def prepare_glyph(gray: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Crop the ink, pad it to a centred square, resize to size x size (uint8)."""
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.size == 0 or gray.max() == 0:
        return np.zeros((size, size), dtype=np.uint8)
    mask = gray > max(30, int(gray.max() * 0.3))
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    crop = gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    side = int(max(h, w) * 1.2) + 2
    canvas = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top: top + h, left: left + w] = crop
    img = Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def rgba_to_gray(rgba: np.ndarray) -> np.ndarray:
    """Canvas RGBA (white stroke on black) -> grey uint8 with bright ink."""
    img = Image.fromarray(np.asarray(rgba).astype("uint8"))
    return np.asarray(img.convert("L"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Lexicon handling: turn a word into the sequence of shapes a writer draws
# ---------------------------------------------------------------------------
BAR = "<BAR>"  # া ি ী ৗ: a tall thin stroke that becomes its own segment
E_SIGN = "<E>"  # ে ৈ: a small loop written before the consonant

HASANTA = "্"
NUKTA = "়"
PRE_SIGNS = {"ি": [BAR], "ে": [E_SIGN], "ৈ": [E_SIGN], "ো": [E_SIGN], "ৌ": [E_SIGN]}
POST_SIGNS = {"া": [BAR], "ী": [BAR], "ো": [BAR], "ৌ": [BAR]}
DROPPED = {"ু", "ূ", "ৃ", "ঁ", "‌", "‍"}  # written under/over a letter: same segment


def word_to_tokens(word: str, label_set: set[str]) -> list[str] | None:
    """Return the left-to-right shape tokens of a word, or None if it cannot be read."""
    word = nfc(word.strip())
    chars = list(word)
    units: list[str] = []
    i = 0
    while i < len(chars):  # glue nukta letters (য় ড় ঢ়) into one unit
        if i + 1 < len(chars) and chars[i + 1] == NUKTA:
            units.append(chars[i] + NUKTA)
            i += 2
        else:
            units.append(chars[i])
            i += 1

    tokens: list[str] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u in DROPPED:
            i += 1
            continue
        if u not in label_set:
            return None
        base = u
        j = i + 1
        # conjunct C1 + hasanta + C2 (+ ...) is written as one glyph: one token
        while j + 1 < len(units) and units[j] == HASANTA and units[j + 1] in label_set:
            if base == "র" and j == i + 1:  # reph is a small mark above the next consonant
                base = units[j + 1]
            j += 2
        if j < len(units) and units[j] == HASANTA:  # a final hasanta
            j += 1
        pre: list[str] = []
        post: list[str] = []
        while j < len(units) and (units[j] in PRE_SIGNS or units[j] in POST_SIGNS or units[j] in DROPPED):
            pre += PRE_SIGNS.get(units[j], [])
            post += POST_SIGNS.get(units[j], [])
            j += 1
        tokens += pre + [base] + post
        i = j
    return tokens or None


def load_lexicon(path: Path = LEXICON_PATH) -> list[str]:
    if not Path(path).exists():
        return []
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = nfc(line.strip())
        if line and not line.startswith("#"):
            words.append(line)
    return list(dict.fromkeys(words))


# ---------------------------------------------------------------------------
# Word segmentation
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    x0: int
    x1: int
    bar_prob: float  # how much the segment looks like a vowel bar (া ি ী)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def segment_word(gray: np.ndarray, stroke_width: int = 10) -> dict:
    """Split a drawn word into character segments.

    1. Find the ink box.
    2. Find the Bangla headline (matra): the longest horizontal ink row in the top part.
    3. Look only below the headline and cut at empty columns.
    """
    ink = gray > 100
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return {"segments": [], "headline": None, "box": None}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    box = ink[y0: y1 + 1, x0: x1 + 1]
    H, W = box.shape

    rows = box.sum(axis=1)
    top_part = max(1, int(H * 0.45))
    r = int(np.argmax(rows[:top_part]))
    headline = None
    body_top = 0
    if rows[r] >= 0.35 * W and W > 2 * stroke_width:
        lo = hi = r
        cap = max(3, stroke_width)  # a headline is about one pen-width thick
        while lo - 1 >= max(0, r - cap) and rows[lo - 1] >= 0.4 * rows[r]:
            lo -= 1
        while hi + 1 < min(H, r + cap + 1) and rows[hi + 1] >= 0.4 * rows[r]:
            hi += 1
        hi = min(H - 1, hi + max(1, stroke_width // 4))
        headline = (int(lo + y0), int(hi + y0))
        body_top = hi + 1

    body = box[body_top:, :]
    if body.size == 0 or body.sum() == 0:
        body = box
        body_top = 0
    cols = body.sum(axis=0) > 0
    runs = _runs(cols)

    # join runs split by a tiny gap and drop specks
    gap_tol = max(1, stroke_width // 5)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= gap_tol:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    min_ink = max(4, int(0.6 * stroke_width * stroke_width))
    merged = [m for m in merged if body[:, m[0]: m[1] + 1].sum() >= min_ink]

    body_h = body.shape[0]

    # letters that touch each other below the headline: cut very wide pieces
    # at their thinnest column (the decoder can glue wrong cuts back together)
    widths = [b - a + 1 for a, b in merged if b - a + 1 > 2.6 * stroke_width]
    typical = float(np.median(widths)) if len(widths) >= 2 else 0.7 * body_h
    typical = min(max(typical, 0.45 * body_h), 0.7 * body_h)
    col_ink = body.sum(axis=0)

    def split_wide(a: int, b: int) -> list[list[int]]:
        w = b - a + 1
        if w <= 1.35 * typical:
            return [[a, b]]
        lo_c, hi_c = a + int(0.3 * w), b - int(0.3 * w)
        cut = lo_c + int(np.argmin(col_ink[lo_c: hi_c + 1]))
        return split_wide(a, cut) + split_wide(cut + 1, b)

    merged = [piece for a, b in merged for piece in split_wide(a, b)]

    segments: list[Segment] = []
    bottoms: list[int] = []
    for a, b in merged:
        part = body[:, a: b + 1]
        rows_with_ink = np.nonzero(part.sum(axis=1))[0]
        h = rows_with_ink.max() - rows_with_ink.min() + 1 if len(rows_with_ink) else 0
        w = b - a + 1
        thin = w <= max(0.4 * h, 2.6 * stroke_width)
        tall = h >= 0.5 * body_h
        bar_prob = 0.85 if (thin and tall) else 0.04
        segments.append(Segment(int(a + x0), int(b + x0), bar_prob))
        if len(rows_with_ink) and h >= 0.45 * body_h:
            bottoms.append(int(rows_with_ink.max() + body_top + y0))

    # baseline: where most letters end. Signs written under a letter (ু ূ ৃ)
    # hang below it and are cut off before the letter is classified.
    baseline = int(np.percentile(bottoms, 25)) if bottoms else int(y1)
    zone = max(1, baseline - (headline[0] if headline else int(y0)))
    cut_y = min(int(y1), baseline + max(stroke_width, int(0.12 * zone)))
    return {"segments": segments, "headline": headline, "box": (int(x0), int(y0), int(x1), int(y1)),
            "baseline": baseline, "cut_y": cut_y}


# ---------------------------------------------------------------------------
# Word recognizer
# ---------------------------------------------------------------------------
LOG_FLOOR = math.log(1e-6)
SKIP_SEGMENT = math.log(0.02)
SKIP_LETTER = math.log(0.003)
SKIP_MARK = math.log(0.05)
MAX_SPAN = 3


class WordRecognizer:
    def __init__(self, model, labels: list[str], lexicon: list[str]):
        self.model = model
        self.labels = labels
        self.label_index = {c: i for i, c in enumerate(labels)}
        self.label_set = set(labels)
        self.set_lexicon(lexicon)

    def set_lexicon(self, words: list[str]) -> None:
        self.lexicon: list[tuple[str, list[str]]] = []
        for w in dict.fromkeys(nfc(x) for x in words if x.strip()):
            toks = word_to_tokens(w, self.label_set)
            if toks:
                self.lexicon.append((w, toks))

    # --- classifier on every span of 1..3 neighbouring segments -------------
    def _span_probs(self, gray: np.ndarray, segments: list[Segment], cut_y: int) -> dict:
        gray = gray[: cut_y + 1]
        crops, keys = [], []
        for s in range(len(segments)):
            for k in range(1, MAX_SPAN + 1):
                if s + k > len(segments):
                    break
                xa, xb = segments[s].x0, segments[s + k - 1].x1
                crops.append(prepare_glyph(gray[:, xa: xb + 1]))
                keys.append((s, k))
        if not crops:
            return {}
        batch = np.stack(crops)[..., None].astype("float32")
        probs = np.asarray(self.model(batch, training=False))
        out = {}
        for key, p, crop in zip(keys, probs, crops):
            out[key] = (np.log(np.clip(p, 1e-6, 1.0)), crop)
        return out

    def _token_cost(self, token: str, s: int, k: int, segments, span) -> float | None:
        if token == BAR:
            return math.log(segments[s].bar_prob) if k == 1 else None
        if token == E_SIGN:
            if k != 1:
                return None
            return math.log(0.25 if segments[s].bar_prob < 0.5 else 0.03)
        logp = span[(s, k)][0]
        score = float(logp[self.label_index[token]])
        if NUKTA in token and token[0] in self.label_index:  # the dot below may be cut off
            score = max(score, float(logp[self.label_index[token[0]]]) - 0.3)
        return score - 0.35 * (k - 1)

    def _align(self, tokens: list[str], segments, span) -> float:
        n, T = len(segments), len(tokens)
        NEG = -1e18
        dp = np.full((T + 1, n + 1), NEG)
        dp[0, 0] = 0.0
        for t in range(T + 1):
            for s in range(n + 1):
                cur = dp[t, s]
                if cur <= NEG / 2:
                    continue
                if s < n:  # leave a segment unexplained
                    dp[t, s + 1] = max(dp[t, s + 1], cur + SKIP_SEGMENT)
                if t < T:  # a token that was not drawn / not found
                    pen = SKIP_MARK if tokens[t] in (BAR, E_SIGN) else SKIP_LETTER
                    dp[t + 1, s] = max(dp[t + 1, s], cur + pen)
                    for k in range(1, MAX_SPAN + 1):
                        if s + k > n:
                            break
                        c = self._token_cost(tokens[t], s, k, segments, span)
                        if c is not None:
                            dp[t + 1, s + k] = max(dp[t + 1, s + k], cur + c)
        return float(dp[T, n])

    def _raw_reading(self, segments, span) -> tuple[str, list[dict]]:
        """Best character reading without a dictionary."""
        n = len(segments)
        best = [(-1e18, None)] * (n + 1)
        best[0] = (0.0, None)
        for s in range(n):
            if best[s][0] <= -1e17:
                continue
            for k in range(1, MAX_SPAN + 1):
                if s + k > n:
                    break
                logp = span[(s, k)][0]
                ci = int(np.argmax(logp))
                cand = [(float(logp[ci]) - 0.35 * (k - 1) - 0.5, self.labels[ci])]
                if k == 1:
                    cand.append((math.log(segments[s].bar_prob), "া"))
                for score, ch in cand:
                    tot = best[s][0] + score
                    if tot > best[s + k][0]:
                        best[s + k] = (tot, (s, k, ch, score))
        parts, s = [], n
        while s > 0 and best[s][1] is not None:
            ps, k, ch, score = best[s][1]
            parts.append({"start": ps, "length": k, "char": ch, "confidence": math.exp(score + 0.5) if ch != "া" else math.exp(score)})
            s = ps
        parts.reverse()
        return nfc("".join(p["char"] for p in parts)), parts

    def recognize(self, gray: np.ndarray, stroke_width: int = 10, use_lexicon: bool = True,
                  top_k: int = 5) -> dict:
        seg = segment_word(gray, stroke_width)
        segments = seg["segments"]
        if not segments:
            return {"word": "", "segmentation": seg, "candidates": [], "raw": "", "parts": [], "crops": []}
        span = self._span_probs(gray, segments, seg["cut_y"])
        raw, parts = self._raw_reading(segments, span)
        crops = [span[(i, 1)][1] for i in range(len(segments))]

        candidates = []
        if use_lexicon and self.lexicon:
            n = len(segments)
            for w, toks in self.lexicon:
                if len(toks) > 2 * n + 2 or len(toks) * MAX_SPAN + 2 < n:
                    continue
                candidates.append((self._align(toks, segments, span), w))
            candidates.sort(reverse=True)
        cand_rows = [{"word": w, "score_per_segment": sc / len(segments)} for sc, w in candidates[:top_k]]

        word, source = raw, "character reading"
        if cand_rows and cand_rows[0]["score_per_segment"] > math.log(0.02):
            word, source = cand_rows[0]["word"], "dictionary"
        return {
            "word": word,
            "source": source,
            "raw": raw,
            "parts": parts,
            "candidates": cand_rows,
            "segmentation": seg,
            "crops": crops,
        }
