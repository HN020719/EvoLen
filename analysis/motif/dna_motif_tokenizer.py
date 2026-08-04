"""Wrapper for the canonical DNAMotifTokenizer (HF: Anonymous-843q0u4q08/DNAMotifTokenizer).

Exposes a `tokenizers.Tokenizer`-compatible interface (.encode, .get_vocab) so
motif_coverage_eval.py can consume it like the other tokenizers.
"""
import itertools
import pickle
import random
from os.path import join
from pathlib import Path


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.features = []


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.lookup_table = {}

    def insert(self, word, features=None):
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]
        current_node.is_end_of_word = True
        if features:
            current_node.features.append(features)

    def search(self, word):
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                return False
            current_node = current_node.children[char]
        if current_node.is_end_of_word:
            if len(current_node.features) > 0:
                return current_node.features
            return True
        return False


class _Encoding:
    def __init__(self, tokens):
        self.tokens = tokens
        self.offsets = None


class DNAMotifTokenizer:
    """Canonical greedy-longest-match tokenizer with +1/+2 lookahead."""

    MAXLEN = 12

    def __init__(self, tokenizer_dir, seed=0):
        tokenizer_dir = Path(tokenizer_dir)
        with open(tokenizer_dir / "motifs_hardcode_trie.pkl", "rb") as f:
            self.trie = pickle.load(f)

        self.k1 = ["A", "T", "C", "G", "N"]
        self.k3 = ["".join(t) for t in itertools.product("ATCG", repeat=3)]

        self.lookup = {}
        with open(tokenizer_dir / "motifs_dedup.txt") as f:
            for line in f:
                seg, name = line.strip().split(maxsplit=1)
                self.lookup[seg] = name

        self.vocab = {}
        with open(tokenizer_dir / "vocab_dedup.txt") as f:
            for i, line in enumerate(f):
                tok = line.strip()
                if tok:
                    self.vocab[tok] = i

        self._rng = random.Random(seed)

    def get_vocab(self):
        return dict(self.vocab)

    def get_vocab_size(self):
        return len(self.vocab)

    def _tokenize_at(self, seg, i):
        candidates = []
        for l in range(4, self.MAXLEN + 1):
            segment = seg[i:i + l]
            if self.trie.search(segment):
                candidates.append(segment)

        if candidates:
            best = self._rng.choice(candidates)
            return best, len(best)

        # Fallback: 3-mer then 1-mer
        for l in range(3, 0, -1):
            segment = seg[i:i + l]
            if l == 3 and segment in self.k3:
                return segment, 3
            if l == 1 and segment in self.k1:
                return segment, 1
        # Last resort: emit single char even if unknown (e.g. 'N' outside k1 — but k1 has N)
        return seg[i:i + 1], 1

    def _tokenize_seq(self, seg):
        tokens = []
        i = 0
        while i < len(seg):
            best_token, best_score = self._tokenize_at(seg, i)
            best_i = i
            next_pos = i + len(best_token)

            offsets = [1, 2] if len(best_token) > 1 else []
            for shift in offsets:
                i_shifted = i + shift
                if i_shifted < len(seg):
                    tok_, score_ = self._tokenize_at(seg, i_shifted)
                    next_pos_ = i_shifted + len(tok_)
                    if score_ > best_score:
                        best_token, best_score, best_i, next_pos = tok_, score_, i_shifted, next_pos_

            for skip in range(best_i - i):
                tokens.append(seg[i + skip])
            tokens.append(best_token)
            i = next_pos
        return tokens

    def encode(self, text, add_special_tokens=False):
        return _Encoding(self._tokenize_seq(text))

    def encode_batch(self, texts, add_special_tokens=False):
        return [_Encoding(self._tokenize_seq(t)) for t in texts]
