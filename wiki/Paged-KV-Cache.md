# Paged KV-Cache

The KV-cache stores the key and value tensors for every token a sequence has seen, so that attention does not recompute them on each step. It is the dominant consumer of memory in LLM serving, and how you allocate it decides how many sequences you can run at once. forge-infer implements the block-based paged allocator that vLLM popularised. This page explains why, and how the implementation in `src/paged_cache.rs` works.

## The problem with contiguous allocation

The obvious design reserves one contiguous buffer per sequence, sized to the maximum context length. It wastes memory two ways:

- **Internal fragmentation.** A sequence that stops after 30 tokens but was sized for 2048 leaves the rest of its buffer idle for its whole lifetime.
- **External fragmentation.** As sequences of different sizes come and go, the free pool fractures into holes. A new request may need 800 contiguous slots while 800 are free in total but scattered across holes none of which is big enough.

Both effects mean you run far fewer concurrent sequences than your memory budget should allow.

## Paging

Split memory into fixed-size **blocks**, each holding `block_size` token slots. Each sequence keeps a **block table**: an ordered list of the physical blocks that store its tokens. Blocks are allocated one at a time, lazily, as the sequence grows. The consequences:

- **External fragmentation disappears.** Every free block is interchangeable, so any free block satisfies any sequence. There are no holes of the wrong shape.
- **Internal fragmentation is bounded.** The only waste is the partially filled final block of each sequence, at most `block_size - 1` slots. With a block size of 16 that is a handful of slots per sequence rather than thousands.

## The allocator

```rust
pub struct PagedKVCache {
    block_size: usize,
    num_blocks: usize,
    free_list: Vec<BlockId>,
    tables: HashMap<SeqId, BlockTable>,
}
```

The free list is a stack of block ids. The map holds one `BlockTable` per live sequence. The key operations:

- `admit(seq)` registers a sequence with an empty table.
- `blocks_needed_for(seq, extra_tokens)` answers, without mutating anything, how many new blocks a sequence would need to store `extra_tokens` more tokens. It accounts for the slack in the current final block. This is the question the scheduler asks before committing a decode step.
- `append(seq, count)` reserves the blocks and records the tokens. It is **transactional**: it pre-checks the free count and returns `OutOfBlocks { needed, free }` without touching state if the cache cannot grow the sequence, so the scheduler can preempt and retry safely.
- `free(seq)` returns every block to the free list and forgets the sequence. It is idempotent.
- `evict_largest()` frees the sequence holding the most blocks, the fallback when even preempting the running batch cannot place a new sequence.

## Lazy growth in detail

A sequence's `BlockTable` reports `capacity = blocks * block_size` and `slack = capacity - num_tokens`. When you append `count` tokens, the allocator only pulls new blocks if `count` exceeds the current slack:

```rust
let slack = table.slack(self.block_size);
let needed = if extra_tokens <= slack {
    0
} else {
    (extra_tokens - slack).div_ceil(self.block_size)
};
```

So the first token of a sequence pulls one block, the next `block_size - 1` tokens reuse it, and only the token that overflows the block pulls another. That is exactly the test `lazy_growth_only_allocates_when_a_block_fills` asserts.

## What the tests prove

The suite in `src/paged_cache.rs` pins down the tricky behaviour:

- `allocate_and_free_round_trips`: freeing returns every block.
- `lazy_growth_only_allocates_when_a_block_fills`: no premature allocation.
- `internal_fragmentation_is_bounded_by_block_size`: waste stays under `block_size` per sequence.
- `out_of_blocks_is_reported_and_leaves_state_intact`: a failed append does not partially grow a sequence.
- `no_external_fragmentation_after_interleaved_free`: a freed middle block is immediately reusable by a larger new sequence, the property a contiguous allocator cannot offer.
- `eviction_picks_the_largest_sequence`: the eviction policy targets the biggest consumer.
- `blocks_needed_accounts_for_slack`: the planning query is exact.

## Tuning

The two knobs are `num_blocks` and `block_size`, set when you construct the cache (the server uses 512 blocks of 16 in `default_state`). Larger blocks mean fewer allocator operations but more internal fragmentation; smaller blocks mean tighter packing but more bookkeeping. A block size of 16 is a sensible middle that mirrors common production settings.
