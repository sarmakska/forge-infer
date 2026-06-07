# Paged KV-Cache

The KV-cache stores the key and value tensors for every token a sequence has seen, so attention does not recompute them each step. It is the dominant consumer of memory in LLM serving, and how you allocate it decides how many sequences you can run at once. forge-infer implements the block-based paged allocator that vLLM popularised, in `src/paged_cache.rs`. This page explains why, how it works, and walks one allocation through by hand.

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
    free_list: Vec<BlockId>,        // a stack; pop from the back keeps freed blocks hot
    tables: HashMap<SeqId, BlockTable>,
    peak_blocks: usize,             // high-water mark of simultaneously used blocks
}
```

The key operations, all in `src/paged_cache.rs`:

- `admit(seq)` registers a sequence with an empty table.
- `blocks_needed_for(seq, extra_tokens)` answers, without mutating anything, how many new blocks a sequence would need to store `extra_tokens` more tokens, accounting for slack in the current final block. This is the question the scheduler asks before committing a decode step.
- `append(seq, count)` reserves the blocks and records the tokens. It is **transactional**: it pre-checks the free count and returns `OutOfBlocks { needed, free }` without touching state if the cache cannot grow the sequence, so the scheduler can preempt and retry safely.
- `free(seq)` returns every block to the free list and forgets the sequence. It is idempotent, so the scheduler can free freely.
- `evict_largest()` frees the sequence holding the most blocks, the fallback when even preempting the running batch cannot place a new sequence.

### Observability

Alongside the live counters (`free_blocks`, `used_blocks`, `utilisation`, `internal_fragmentation`) the cache records a lifetime high-water mark:

- `peak_blocks()` returns the largest number of blocks ever held simultaneously. It only ever moves inside `append`, since allocation is the only operation that grows usage, so tracking it costs one `max` per successful append and nothing else. This is the true minimum cache size the workload needed: a run that peaked at `peak_blocks` would behave identically in a cache sized to exactly that many. The benchmark surfaces it as `peak_kv_blocks` to make the memory argument for paging concrete, see [Benchmarks](Benchmarks).

## Lazy growth in detail

A sequence's `BlockTable` reports `capacity = blocks * block_size` and `slack = capacity - num_tokens`. When you append `count` tokens, the allocator pulls new blocks only if `count` exceeds the current slack:

```rust
let slack = table.slack(self.block_size);
let needed = if extra_tokens <= slack {
    0
} else {
    (extra_tokens - slack).div_ceil(self.block_size)
};
```

So the first token of a sequence pulls one block, the next `block_size - 1` tokens reuse it, and only the token that overflows the block pulls another.

## Worked example

A cache of 8 blocks, block size 4, one sequence growing token by token (this is `lazy_growth_only_allocates_when_a_block_fills`):

| Call | num_tokens | blocks held | free blocks | why |
| --- | ---: | ---: | ---: | --- |
| `admit(1)` | 0 | 0 | 8 | empty table |
| `append(1, 1)` | 1 | 1 | 7 | first token pulls a block, 3 slack |
| `append(1, 3)` | 4 | 1 | 7 | fills the slack, no new block |
| `append(1, 1)` | 5 | 2 | 6 | fifth token spills into a second block |
| `free(1)` | gone | 0 | 8 | every block returns |

After `free`, the freed blocks land back on the stack and the next sequence reuses them immediately. A contiguous allocator would leave a hole here that a larger request could not use; `no_external_fragmentation_after_interleaved_free` demonstrates exactly that case by freeing a middle sequence and placing a larger one in its blocks.

## Failure modes

- **Out of blocks.** `append` returns `CacheError::OutOfBlocks { needed, free }` and leaves the sequence untouched. The scheduler turns this into a preemption decision rather than failing the request. `out_of_blocks_is_reported_and_leaves_state_intact` asserts the failed sequence did not grow and no blocks leaked.
- **Unknown sequence.** Appending to a sequence that was never admitted returns `CacheError::UnknownSequence(seq)` rather than panicking.
- **Block size zero.** `new` asserts `block_size > 0`; a zero block size is a programmer error, not a runtime condition.

## What the tests prove

The suite in `src/paged_cache.rs` pins down the tricky behaviour: `allocate_and_free_round_trips`, `lazy_growth_only_allocates_when_a_block_fills`, `internal_fragmentation_is_bounded_by_block_size`, `out_of_blocks_is_reported_and_leaves_state_intact`, `no_external_fragmentation_after_interleaved_free`, `eviction_picks_the_largest_sequence`, `blocks_needed_accounts_for_slack`, `unknown_sequence_append_errors` and `peak_blocks_tracks_the_high_water_mark_not_the_current_use`, which checks that the peak holds when blocks are freed and does not fall on a later, smaller burst.

## Tuning

The two knobs are `num_blocks` and `block_size`, set when you construct the cache (the server uses 512 blocks of 16 in `default_state`). Larger blocks mean fewer allocator operations but more internal fragmentation; smaller blocks mean tighter packing but more bookkeeping. A block size of 16 is a sensible middle that mirrors common production settings.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
