import os
import glob
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

COMPANY_DIRS = {
    "hackerrank": str(DATA_DIR / "hackerrank"),
    "claude": str(DATA_DIR / "claude"),
    "visa": str(DATA_DIR / "visa"),
}


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].strip()
    return text.strip()


def _load_doc(path: str) -> tuple[str, str]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except OSError:
        return path, ""
    content = _strip_frontmatter(raw)
    # Include the filename slug as extra signal for matching
    slug = Path(path).stem.replace("-", " ").replace("_", " ")
    return path, f"{slug}\n{content}"


class Retriever:
    def __init__(self):
        self._paths: list[str] = []
        self._texts: list[str] = []
        self._company_indices: dict[str, list[int]] = {c: [] for c in COMPANY_DIRS}
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._build()

    def _build(self):
        all_md = glob.glob(str(DATA_DIR / "**" / "*.md"), recursive=True)
        for path in all_md:
            p, text = _load_doc(path)
            if not text:
                continue
            idx = len(self._paths)
            self._paths.append(p)
            self._texts.append(text)
            for company, cdir in COMPANY_DIRS.items():
                if p.startswith(cdir):
                    self._company_indices[company].append(idx)

        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._matrix = self._vectorizer.fit_transform(self._texts)

    @staticmethod
    def _clean_snippet(text: str, max_chars: int = 380) -> str:
        if len(text) <= max_chars:
            return text.strip()
        chunk = text[:max_chars]
        for sep in (". ", ".\n", "\n\n", "\n"):
            pos = chunk.rfind(sep)
            if pos > max_chars // 2:
                return chunk[:pos + 1].strip()
        return chunk.strip()

    def retrieve(self, query: str, company: str | None, top_k: int = 2) -> list[dict]:
        q_vec = self._vectorizer.transform([query])
        company_key = (company or "").lower()

        if company_key in self._company_indices and self._company_indices[company_key]:
            indices = self._company_indices[company_key]
            sub_matrix = self._matrix[indices]
            sims = cosine_similarity(q_vec, sub_matrix).flatten()
            top_local = np.argsort(sims)[::-1][:top_k]
            results = [(indices[i], sims[i]) for i in top_local if sims[i] > 0]
        else:
            sims = cosine_similarity(q_vec, self._matrix).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            results = [(i, sims[i]) for i in top_idx if sims[i] > 0]

        docs = []
        for idx, score in results:
            snippet = self._clean_snippet(self._texts[idx])
            docs.append({
                "path": self._paths[idx],
                "score": float(score),
                "snippet": snippet,
            })
        return docs
