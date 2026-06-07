//! A paged key/value cache, modelled on the block-based allocator that vLLM
//! popularised.
//!
//! ## Why paging
//!
//! A naive KV-cache reserves one contiguous buffer per sequence sized to the
//! maximum context length. That wastes memory in two ways: a sequence that
//! stops early leaves the tail of its buffer unused (internal fragmentation),
//! and the pool of free memory gets carved into oddly sized holes that no new
//! sequence fits into (external fragmentation).
//!
//! Paging fixes both. Memory is split into fixed-size blocks. Each sequence
//! holds a *block table*, an ordered list of the physical blocks that store its
//! tokens. Blocks are allocated lazily, one at a time, as the sequence grows.
//! Any free block fits any sequence, so external fragmentation disappears, and
//! the only internal waste is the partially filled final block of each
//! sequence, bounded by the block size.
//!
//! This module implements that allocator for real: a free list, per-sequence
//! block tables, lazy growth, append, free, and the out-of-blocks signal the
//! scheduler needs in order to preempt.

use std::collections::HashMap;

/// Identifies a sequence to the cache. The scheduler owns these.
pub type SeqId = u64;

/// A physical block index into the cache.
pub type BlockId = usize;

/// Returned when the cache cannot satisfy an allocation. The scheduler turns
/// this into a preemption decision rather than failing the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// No free blocks remain. Carries how many more blocks were needed.
    OutOfBlocks { needed: usize, free: usize },
    /// The sequence id was not known to the cache.
    UnknownSequence(SeqId),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::OutOfBlocks { needed, free } => {
                write!(f, "out of KV blocks: needed {needed}, {free} free")
            }
            CacheError::UnknownSequence(s) => write!(f, "unknown sequence {s}"),
        }
    }
}

impl std::error::Error for CacheError {}

/// The block table for one sequence: which physical blocks hold its tokens, and
/// how many tokens are stored in total.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
    pub num_tokens: usize,
}

impl BlockTable {
    /// The number of token slots currently reserved (block count times block
    /// size). Always greater than or equal to `num_tokens`.
    pub fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }

    /// Slots reserved but not yet filled in the final block. This is the only
    /// internal fragmentation paging cannot avoid, and it is bounded by
    /// `block_size - 1`.
    pub fn slack(&self, block_size: usize) -> usize {
        self.capacity(block_size) - self.num_tokens
    }
}

/// The block-based KV-cache allocator.
pub struct PagedKVCache {
    block_size: usize,
    num_blocks: usize,
    free_list: Vec<BlockId>,
    tables: HashMap<SeqId, BlockTable>,
    /// The largest number of blocks ever simultaneously in use. This is the
    /// real peak KV memory the workload demanded, the headline number paging is
    /// supposed to keep low: it never exceeds `num_blocks`, and under continuous
    /// batching it stays far below the sum of per-sequence worst cases.
    peak_blocks: usize,
}

impl PagedKVCache {
    /// Create a cache with `num_blocks` blocks of `block_size` token slots each.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "block size must be positive");
        // The free list is a stack. Popping from the back keeps recently freed
        // blocks hot, which is friendly to a real allocator's locality.
        let free_list = (0..num_blocks).collect();
        Self {
            block_size,
            num_blocks,
            free_list,
            tables: HashMap::new(),
            peak_blocks: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    pub fn used_blocks(&self) -> usize {
        self.num_blocks - self.free_list.len()
    }

    /// The high-water mark of simultaneously used blocks across the cache's
    /// lifetime. A workload that fits in `peak_blocks` blocks would run
    /// unchanged in a cache sized to exactly that many, so this is the true
    /// minimum capacity the run needed.
    pub fn peak_blocks(&self) -> usize {
        self.peak_blocks
    }

    /// Fraction of physical token slots that are reserved by some sequence.
    pub fn utilisation(&self) -> f32 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.used_blocks() as f32 / self.num_blocks as f32
    }

    /// Register a new sequence with an empty block table.
    pub fn admit(&mut self, seq: SeqId) {
        self.tables.entry(seq).or_default();
    }

    /// Whether the sequence is known to the cache.
    pub fn contains(&self, seq: SeqId) -> bool {
        self.tables.contains_key(&seq)
    }

    /// The number of additional blocks a sequence would need to hold
    /// `extra_tokens` more tokens beyond what it currently stores. This is the
    /// question the scheduler asks before committing a decode step.
    pub fn blocks_needed_for(&self, seq: SeqId, extra_tokens: usize) -> usize {
        let table = match self.tables.get(&seq) {
            Some(t) => t,
            None => {
                // A fresh sequence needs ceil(extra / block_size) blocks.
                return extra_tokens.div_ceil(self.block_size);
            }
        };
        let slack = table.slack(self.block_size);
        if extra_tokens <= slack {
            0
        } else {
            (extra_tokens - slack).div_ceil(self.block_size)
        }
    }

    /// Append `count` tokens to a sequence, allocating physical blocks as
    /// needed. Returns `OutOfBlocks` if the cache cannot grow the sequence,
    /// leaving the sequence untouched so the caller can preempt and retry.
    pub fn append(&mut self, seq: SeqId, count: usize) -> Result<(), CacheError> {
        if !self.tables.contains_key(&seq) {
            return Err(CacheError::UnknownSequence(seq));
        }
        let needed = self.blocks_needed_for(seq, count);
        if needed > self.free_list.len() {
            return Err(CacheError::OutOfBlocks {
                needed,
                free: self.free_list.len(),
            });
        }
        // Allocation cannot fail past this point because we pre-checked.
        for _ in 0..needed {
            let block = self.free_list.pop().expect("pre-checked free count");
            let table = self.tables.get_mut(&seq).expect("pre-checked membership");
            table.blocks.push(block);
        }
        let table = self.tables.get_mut(&seq).expect("pre-checked membership");
        table.num_tokens += count;
        // Record the high-water mark. Allocation only ever grows usage, so the
        // peak can only move here.
        self.peak_blocks = self.peak_blocks.max(self.used_blocks());
        Ok(())
    }

    /// Return a sequence's blocks to the free list and forget it. Idempotent:
    /// freeing an unknown sequence is a no-op so the scheduler can free freely.
    pub fn free(&mut self, seq: SeqId) {
        if let Some(table) = self.tables.remove(&seq) {
            for block in table.blocks {
                self.free_list.push(block);
            }
        }
    }

    /// The block table for a sequence, if it exists. The forward pass uses this
    /// to gather the physical KV blocks for attention.
    pub fn block_table(&self, seq: SeqId) -> Option<&BlockTable> {
        self.tables.get(&seq)
    }

    /// How many tokens are stored for a sequence.
    pub fn seq_len(&self, seq: SeqId) -> usize {
        self.tables.get(&seq).map(|t| t.num_tokens).unwrap_or(0)
    }

    /// Internal fragmentation across all live sequences: token slots reserved
    /// in final blocks that hold no token yet. Useful for the benchmark and for
    /// asserting that paging keeps waste bounded.
    pub fn internal_fragmentation(&self) -> usize {
        self.tables.values().map(|t| t.slack(self.block_size)).sum()
    }

    /// Evict the sequence holding the most blocks to make room, returning its
    /// id. This is the policy the scheduler falls back on when even after
    /// preempting the running batch it cannot place a new sequence. Returns
    /// `None` when there is nothing to evict.
    pub fn evict_largest(&mut self) -> Option<SeqId> {
        let victim = self
            .tables
            .iter()
            .max_by_key(|(_, t)| t.blocks.len())
            .map(|(id, _)| *id)?;
        self.free(victim);
        Some(victim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free_round_trips() {
        let mut cache = PagedKVCache::new(8, 4);
        assert_eq!(cache.free_blocks(), 8);

        cache.admit(1);
        cache.append(1, 10).unwrap(); // ceil(10/4) = 3 blocks
        assert_eq!(cache.used_blocks(), 3);
        assert_eq!(cache.seq_len(1), 10);

        cache.free(1);
        assert_eq!(cache.free_blocks(), 8, "freeing must return every block");
        assert!(!cache.contains(1));
    }

    #[test]
    fn lazy_growth_only_allocates_when_a_block_fills() {
        let mut cache = PagedKVCache::new(8, 4);
        cache.admit(1);

        cache.append(1, 1).unwrap();
        assert_eq!(cache.used_blocks(), 1, "first token needs one block");

        // Three more tokens fit in the same block, no new allocation.
        cache.append(1, 3).unwrap();
        assert_eq!(cache.used_blocks(), 1);

        // The fifth token spills into a second block.
        cache.append(1, 1).unwrap();
        assert_eq!(cache.used_blocks(), 2);
    }

    #[test]
    fn internal_fragmentation_is_bounded_by_block_size() {
        let mut cache = PagedKVCache::new(16, 4);
        for s in 0..3u64 {
            cache.admit(s);
            cache.append(s, 5).unwrap(); // 2 blocks, 8 slots, 5 used -> 3 slack
        }
        // Each sequence wastes at most block_size - 1 slots.
        assert!(cache.internal_fragmentation() <= 3 * (4 - 1));
        assert_eq!(cache.internal_fragmentation(), 3 * 3);
    }

    #[test]
    fn out_of_blocks_is_reported_and_leaves_state_intact() {
        let mut cache = PagedKVCache::new(2, 4);
        cache.admit(1);
        cache.append(1, 8).unwrap(); // uses both blocks exactly

        cache.admit(2);
        let err = cache.append(2, 1).unwrap_err();
        assert_eq!(err, CacheError::OutOfBlocks { needed: 1, free: 0 });
        // The failed sequence must not have grown.
        assert_eq!(cache.seq_len(2), 0);
        assert_eq!(cache.used_blocks(), 2);
    }

    #[test]
    fn no_external_fragmentation_after_interleaved_free() {
        // Allocate three sequences, free the middle one, and show its blocks
        // are immediately reusable by a new sequence. A contiguous allocator
        // would leave a hole that a larger request could not use.
        let mut cache = PagedKVCache::new(6, 4);
        for s in 1..=3u64 {
            cache.admit(s);
            cache.append(s, 4).unwrap();
        }
        assert_eq!(cache.free_blocks(), 3);

        cache.free(2); // returns one block to the middle of the pool
        assert_eq!(cache.free_blocks(), 4);

        cache.admit(4);
        // The new sequence happily uses the freed block plus a spare.
        cache.append(4, 8).unwrap();
        assert_eq!(cache.seq_len(4), 8);
        assert_eq!(cache.free_blocks(), 2);
    }

    #[test]
    fn eviction_picks_the_largest_sequence() {
        let mut cache = PagedKVCache::new(16, 4);
        cache.admit(1);
        cache.append(1, 4).unwrap(); // 1 block
        cache.admit(2);
        cache.append(2, 12).unwrap(); // 3 blocks
        cache.admit(3);
        cache.append(3, 8).unwrap(); // 2 blocks

        let victim = cache.evict_largest().unwrap();
        assert_eq!(victim, 2, "the 3-block sequence must be evicted first");
        assert!(!cache.contains(2));
        assert_eq!(cache.free_blocks(), 16 - 1 - 2);
    }

    #[test]
    fn blocks_needed_accounts_for_slack() {
        let mut cache = PagedKVCache::new(16, 4);
        cache.admit(1);
        cache.append(1, 2).unwrap(); // 1 block, 2 slack
        assert_eq!(cache.blocks_needed_for(1, 2), 0, "fits in slack");
        assert_eq!(cache.blocks_needed_for(1, 3), 1, "spills by one token");
        assert_eq!(cache.blocks_needed_for(1, 6), 1);
        assert_eq!(cache.blocks_needed_for(1, 7), 2);
    }

    #[test]
    fn unknown_sequence_append_errors() {
        let mut cache = PagedKVCache::new(4, 4);
        assert_eq!(cache.append(99, 1), Err(CacheError::UnknownSequence(99)));
    }

    #[test]
    fn peak_blocks_tracks_the_high_water_mark_not_the_current_use() {
        // Two sequences allocate three blocks at once, then one is freed. The
        // current use drops but the recorded peak holds: paging let the run fit
        // in three blocks even though only one is live at the end.
        let mut cache = PagedKVCache::new(8, 4);
        assert_eq!(cache.peak_blocks(), 0);

        cache.admit(1);
        cache.append(1, 8).unwrap(); // 2 blocks
        cache.admit(2);
        cache.append(2, 4).unwrap(); // 1 block, 3 in use
        assert_eq!(cache.used_blocks(), 3);
        assert_eq!(cache.peak_blocks(), 3);

        cache.free(1); // 1 block in use now
        assert_eq!(cache.used_blocks(), 1);
        assert_eq!(
            cache.peak_blocks(),
            3,
            "peak must not fall when blocks free"
        );

        // A later, smaller burst does not lower the recorded peak.
        cache.append(2, 4).unwrap();
        assert_eq!(cache.peak_blocks(), 3);
    }
}
