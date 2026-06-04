"""A keyless, offline implementation of the LLM boundary (ADR 0014).

Implements :class:`~cbt_core.llm.client.LlmClient` with no network and no API key, so the whole
platform - ingestion, indexing, and retrieval-augmented question answering - can run for a local
demo without a Gemini key:

- ``analyze_tone`` scores tone with the supervised classifier (ADR 0013) and writes a deterministic
  extractive summary and an honest rationale naming the offline model;
- ``embed`` produces a deterministic hashing embedding (signed feature hashing into
  :data:`~cbt_core.domain.qa.EMBEDDING_DIM` dimensions), good enough for nearest-neighbour
  retrieval; and
- ``answer`` returns an explicitly **extractive** answer assembled from the retrieved passages and
  labelled as such. It never writes prose as if a model generated it: with no LLM, an honest
  extractive answer is the right behaviour, not a silent fabrication (CLAUDE.md section 3).

The Gemini path (ADR 0007) remains the production signal; this client is the offline fallback the
demo runner wires in when no key is present.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from cbt_core.analysis.classifier import ToneClassifier, ngrams, tokenize
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk
from cbt_core.domain.tone import ToneLabel
from cbt_core.llm.client import Embedding

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_LABEL_TO_TONE = {
    "hawkish": ToneLabel.HAWKISH,
    "dovish": ToneLabel.DOVISH,
    "neutral": ToneLabel.NEUTRAL,
}
_EXTRACTIVE_PREFIX = (
    "No language model is configured (offline mode), so this is an extractive answer drawn from "
    "the most relevant retrieved passages:"
)


def _sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [match.group().strip() for match in _SENTENCE_RE.finditer(text) if match.group().strip()]


def _stable_hash(token: str) -> int:
    """A process-independent hash of a token (Python's ``hash`` is salted per run)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class OfflineLlmClient:
    """An offline, keyless LLM boundary: classifier tone, hashing embeddings, extractive answers."""

    def __init__(
        self,
        classifier: ToneClassifier | None = None,
        *,
        dim: int = EMBEDDING_DIM,
        ngram_max: int = 2,
        summary_sentences: int = 3,
    ) -> None:
        """Build the offline client.

        Args:
            classifier: The trained tone classifier; the bundled model is loaded if not supplied.
            dim: The embedding dimensionality (must match the platform's ``EMBEDDING_DIM``).
            ngram_max: The largest n-gram used when hashing text into embeddings.
            summary_sentences: How many sentences the extractive summary keeps.
        """
        self._classifier = classifier if classifier is not None else ToneClassifier.load_default()
        self._dim = dim
        self._ngram_max = ngram_max
        self._summary_sentences = summary_sentences

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        """Score tone with the classifier and build an extractive summary and honest rationale.

        Args:
            speech_text: The full text of the speech.

        Returns:
            A :class:`ToneAnalysis` from the offline classifier (never the MIXED label, which the
            three-class classifier does not produce).
        """
        verdict = self._classifier.score(speech_text)
        probs = verdict.probabilities
        rationale = (
            f"Offline supervised classifier (no LLM): P(hawkish)={probs['hawkish']:.2f}, "
            f"P(dovish)={probs['dovish']:.2f}, P(neutral)={probs['neutral']:.2f}."
        )
        return ToneAnalysis(
            summary=self._summarize(speech_text),
            tone=_LABEL_TO_TONE[verdict.label],
            score=verdict.score,
            rationale=rationale,
        )

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts with deterministic signed feature hashing.

        Args:
            texts: The texts to embed.

        Returns:
            One L2-normalized vector of dimension ``dim`` per input text, in order.
        """
        return [self._embed_one(text) for text in texts]

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Assemble an explicitly-extractive answer from the most relevant retrieved sentences.

        Args:
            question: The user's question.
            chunks: The retrieved context to ground the answer in.

        Returns:
            A labelled extractive answer: the passages closest to the question, with their source
            titles, never generated prose.
        """
        query = np.asarray(self._embed_one(question))
        ranked: list[tuple[float, str, str]] = []
        for chunk in chunks:
            for sentence in _sentences(chunk.text):
                if len(sentence.split()) < 4:
                    continue
                similarity = float(np.asarray(self._embed_one(sentence)) @ query)
                ranked.append((similarity, sentence, chunk.title))
        ranked.sort(key=lambda item: item[0], reverse=True)

        seen: set[str] = set()
        lines: list[str] = []
        for _, sentence, title in ranked:
            if sentence in seen:
                continue
            seen.add(sentence)
            lines.append(f"- {sentence} ({title})")
            if len(lines) >= self._summary_sentences:
                break
        if not lines:
            # Retrieval returned chunks but no scorable sentence; fall back to the chunk bodies.
            lines = [f"- {chunk.text.strip()} ({chunk.title})" for chunk in chunks[:1]]
        return _EXTRACTIVE_PREFIX + "\n\n" + "\n".join(lines)

    def _summarize(self, text: str) -> str:
        """Extractive summary: the lead sentence plus the most tonally salient remaining ones."""
        sentences = _sentences(text)
        if not sentences:
            return text.strip() or "(no text)"
        if len(sentences) <= self._summary_sentences:
            return " ".join(sentences)
        salience = [abs(self._classifier.score(sentence).score) for sentence in sentences]
        # Always keep the lead sentence; fill the rest by descending tonal salience, then restore
        # the original reading order so the summary still reads coherently.
        chosen = {0}
        for index in sorted(range(1, len(sentences)), key=lambda i: salience[i], reverse=True):
            if len(chosen) >= self._summary_sentences:
                break
            chosen.add(index)
        return " ".join(sentences[i] for i in sorted(chosen))

    def _embed_one(self, text: str) -> Embedding:
        """Signed feature hashing of one text into an L2-normalized vector."""
        vec = np.zeros(self._dim, dtype=np.float64)
        for feature in ngrams(tokenize(text), self._ngram_max):
            digest = _stable_hash(feature)
            vec[digest % self._dim] += 1.0 if (digest >> 33) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()
