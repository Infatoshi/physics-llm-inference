from dataclasses import dataclass, field


@dataclass
class RadixNode:
    token_ids: list[int] = field(default_factory=list)
    children: dict[int, 'RadixNode'] = field(default_factory=dict)
    ref_count: int = 0
    kv_indices: list[int] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixCache:
    def __init__(self, block_size: int = 16):
        self.root = RadixNode()
        self.block_size = block_size
        self.total_cached_tokens = 0

    def insert(
        self,
        token_ids: list[int],
        kv_indices: list[int],
    ) -> int:
        if not token_ids:
            return 0

        node = self.root
        tokens_inserted = 0
        idx = 0

        while idx < len(token_ids):
            token = token_ids[idx]

            if token in node.children:
                child = node.children[token]
                match_len = 0
                while (match_len < len(child.token_ids) and
                       idx + match_len < len(token_ids) and
                       child.token_ids[match_len] == token_ids[idx + match_len]):
                    match_len += 1

                if match_len < len(child.token_ids):
                    new_node = RadixNode(
                        token_ids=child.token_ids[match_len:],
                        children=child.children,
                        ref_count=child.ref_count,
                        kv_indices=child.kv_indices[match_len:],
                    )
                    child.token_ids = child.token_ids[:match_len]
                    child.kv_indices = child.kv_indices[:match_len]
                    child.children = {child.token_ids[-1] if child.token_ids else new_node.token_ids[0]: new_node}

                idx += match_len
                node = child
            else:
                remaining_tokens = token_ids[idx:]
                remaining_kv = kv_indices[idx:] if idx < len(kv_indices) else []

                new_node = RadixNode(
                    token_ids=remaining_tokens,
                    kv_indices=remaining_kv,
                )
                node.children[token] = new_node
                tokens_inserted = len(remaining_tokens)
                self.total_cached_tokens += tokens_inserted
                break

        return tokens_inserted

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        if not token_ids:
            return 0, []

        node = self.root
        matched = 0
        kv_indices = []
        idx = 0

        while idx < len(token_ids) and node.children:
            token = token_ids[idx]
            if token not in node.children:
                break

            child = node.children[token]
            match_len = 0
            while (match_len < len(child.token_ids) and
                   idx + match_len < len(token_ids) and
                   child.token_ids[match_len] == token_ids[idx + match_len]):
                match_len += 1
                if match_len <= len(child.kv_indices):
                    kv_indices.append(child.kv_indices[match_len - 1])

            matched += match_len
            idx += match_len

            if match_len == len(child.token_ids):
                node = child
            else:
                break

        return matched, kv_indices

    def get_cache_hit_rate(
        self,
        queries: list[list[int]],
    ) -> float:
        total_tokens = 0
        matched_tokens = 0

        for query in queries:
            total_tokens += len(query)
            matched, _ = self.match_prefix(query)
            matched_tokens += matched

        return matched_tokens / total_tokens if total_tokens > 0 else 0.0


def explain_radix_cache() -> str:
    return """
Radix Cache (Prefix Sharing)

Problem: Many requests share common prefixes
  - System prompts (same for all users)
  - Few-shot examples
  - Conversation history

Without sharing: recompute KV cache for each request
With radix cache: share computed KV values for common prefixes

Radix Tree (Trie):
  - Each path from root to leaf represents a token sequence
  - Nodes store token IDs and corresponding KV cache indices
  - Prefix lookup in O(prefix_length) time

Example:
  Request 1: "You are a helpful assistant. What is 2+2?"
  Request 2: "You are a helpful assistant. Explain quantum physics."
  Request 3: "Hello, how are you?"

Radix tree:
  root
  ├── "You are a helpful assistant. " -> shared KV
  │   ├── "What is 2+2?" -> request 1 KV
  │   └── "Explain quantum physics." -> request 2 KV
  └── "Hello, how are you?" -> request 3 KV

Benefits:
  - Prefill only the non-shared suffix
  - Significant speedup for system prompts
  - Memory savings from KV cache sharing
"""


if __name__ == "__main__":
    print(explain_radix_cache())

    print("\n" + "=" * 60)
    print("Radix Cache Demo")
    print("-" * 60)

    cache = RadixCache()

    prompt1 = [1, 2, 3, 4, 5, 6, 7, 8]
    kv1 = list(range(100, 108))
    inserted = cache.insert(prompt1, kv1)
    print(f"Inserted prompt1: {inserted} new tokens")

    prompt2 = [1, 2, 3, 4, 10, 11, 12]
    matched, kv_indices = cache.match_prefix(prompt2)
    print(f"Prompt2 matched {matched} tokens, KV indices: {kv_indices[:matched]}")

    prompt3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    matched, kv_indices = cache.match_prefix(prompt3)
    print(f"Prompt3 matched {matched} tokens")

    prompt4 = [20, 21, 22]
    matched, _ = cache.match_prefix(prompt4)
    print(f"Prompt4 matched {matched} tokens (different prefix)")

    print(f"\nTotal cached tokens: {cache.total_cached_tokens}")

    queries = [prompt1, prompt2, prompt3, prompt4]
    hit_rate = cache.get_cache_hit_rate(queries)
    print(f"Cache hit rate: {hit_rate:.1%}")
