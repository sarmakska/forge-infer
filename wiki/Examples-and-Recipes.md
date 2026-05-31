# Examples and Recipes

Copy-paste recipes for driving forge-infer, both over HTTP and as a Rust library. Every example here runs against the code as shipped; the model is deterministic, so the outputs are reproducible byte streams rather than prose (see [Troubleshooting](Troubleshooting) if that surprises you).

## Start the server

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run --release --bin forge-infer            # binds 127.0.0.1:8080
FORGE_ADDR=127.0.0.1:9000 cargo run --release --bin forge-infer   # custom port
RUST_LOG=forge_infer=debug cargo run --bin forge-infer            # verbose logs
```

## HTTP recipes

### Health check

```bash
curl -s localhost:8080/healthz
# {"status":"ok"}
```

### Native completion

```bash
curl -s localhost:8080/generate \
  -d '{"prompt":"hello forge","max_tokens":24}'
# {"text":"...","prompt_tokens":11,"completion_tokens":24}
```

### OpenAI-compatible, non-streaming

```bash
curl -s localhost:8080/v1/completions \
  -d '{"model":"forge-infer","prompt":"the quick brown fox","max_tokens":16}'
# {"id":"cmpl-forge","object":"text_completion","model":"forge-infer",
#  "choices":[{"text":"...","index":0,"finish_reason":"stop"}]}
```

### OpenAI-compatible, streamed over SSE

```bash
curl -sN localhost:8080/v1/completions \
  -d '{"prompt":"stream me","max_tokens":20,"stream":true}'
# data: {"id":"cmpl-forge",...,"choices":[{"text":"s","index":0,"finish_reason":null}]}
# data: {"id":"cmpl-forge",...,"choices":[{"text":"t","index":0,"finish_reason":null}]}
# ...
# data: [DONE]
```

The `-N` flag is essential; without it curl buffers the stream and you see it all at once.

### Point the OpenAI Python client at forge-infer

The `/v1/completions` endpoint matches the OpenAI completion shape, so the official client works with only a `base_url` change:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

resp = client.completions.create(
    model="forge-infer",
    prompt="hello forge",
    max_tokens=16,
)
print(resp.choices[0].text)

# streaming
for chunk in client.completions.create(
    model="forge-infer", prompt="stream me", max_tokens=20, stream=True,
):
    print(chunk.choices[0].text, end="", flush=True)
```

forge-infer ignores the API key (there is no auth; see [Security-Model](Security-Model)), so any non-empty string works.

## Library recipes

Add the crate path-wise or as a git dependency, then drive the pieces directly.

### Run one prompt through the engine

```rust
use forge_infer::engine::Engine;
use forge_infer::model::{Model, TinyTransformer};
use forge_infer::paged_cache::PagedKVCache;
use forge_infer::scheduler::SchedulerConfig;
use forge_infer::tokeniser::{self, EOS_TOKEN, VOCAB_SIZE};
use std::sync::Arc;

let model: Arc<dyn Model> =
    Arc::new(TinyTransformer::new(VOCAB_SIZE, 6).with_eos(EOS_TOKEN));
let mut eng = Engine::new(
    model,
    SchedulerConfig { max_batch_size: 8 },
    PagedKVCache::new(512, 16),
);
eng.submit(1, tokeniser::encode("hello forge"), 32);
for out in eng.run_to_completion() {
    println!("{}", tokeniser::decode(&out.tokens));
}
```

### Drive continuous batching by hand

Submit many requests to one engine and step the loop yourself, watching tokens stream out per iteration:

```rust
let mut eng = Engine::new(model, SchedulerConfig { max_batch_size: 16 },
                          PagedKVCache::new(2048, 16));
for id in 0..64 {
    eng.submit(id, tokeniser::encode("prompt"), 64);
}
while eng.scheduler_mut().has_work() {
    for (id, token, is_eos) in eng.step() {
        // id produced `token` this iteration; is_eos marks the last one
        let _ = (id, token, is_eos);
    }
}
```

This is exactly what the benchmark's continuous path does (`src/bin/bench.rs`).

### Speculative decoding for one stream

```rust
use forge_infer::speculative::SpeculativeDecoder;

let target = TinyTransformer::new(VOCAB_SIZE, 6).with_eos(EOS_TOKEN);
let draft  = TinyTransformer::draft(VOCAB_SIZE, 6).with_eos(EOS_TOKEN);
let dec = SpeculativeDecoder::new(&draft, &target, 4);   // lookahead 4

let (tokens, acceptance_rate) = dec.generate(&tokeniser::encode("hello"), 64);
println!("{} tokens, acceptance {:.0}%", tokens.len(), acceptance_rate * 100.0);
```

### Inspect a single speculative round

```rust
let result = dec.step(&tokeniser::encode("hello forge"));
println!("accepted {}/{} ({:.0}%)",
    result.accepted, result.drafted, result.acceptance_rate() * 100.0);
// result.tokens always has at least one token, even on a full rejection
```

### Watch the cache directly

```rust
use forge_infer::paged_cache::PagedKVCache;

let mut cache = PagedKVCache::new(8, 4);
cache.admit(1);
cache.append(1, 1).unwrap();
println!("used {} free {}", cache.used_blocks(), cache.free_blocks()); // used 1 free 7
cache.append(1, 3).unwrap();   // fills the block, no new allocation
println!("used {}", cache.used_blocks());                              // used 1
cache.append(1, 1).unwrap();   // spills into a second block
println!("used {}", cache.used_blocks());                             // used 2
println!("internal frag {}", cache.internal_fragmentation());          // wasted slots
```

This mirrors the worked example on [Paged-KV-Cache](Paged-KV-Cache).

### Reproduce the preemption scenario

```rust
use forge_infer::scheduler::{Scheduler, SchedulerConfig, Sequence};

// block_size 1 makes every token cost a block, so pressure is exact
let mut sched = Scheduler::new(SchedulerConfig { max_batch_size: 8 },
                               PagedKVCache::new(6, 1));
sched.submit(Sequence::new(1, vec![1, 2], 50));
sched.submit(Sequence::new(2, vec![3, 4], 50));
sched.schedule();                       // both admitted, 4/6 blocks used
sched.push_token(1, 5, false);
sched.push_token(2, 6, false);          // now 6/6 used, 0 free
let plan = sched.schedule();            // forces a preemption
assert_eq!(plan.preempted, vec![2]);    // least-progressed by tie-break
```

## Run the benchmark

```bash
cargo run --release --bin forge-bench
```

Always release mode. See [Benchmarks](Benchmarks) for how to read the output table and the real numbers from an M3 Pro.

## See also

- [HTTP-Server](HTTP-Server) for the endpoint reference.
- [API-Reference](API-Reference) for every public signature.
- [Configuration-and-Tuning](Configuration-and-Tuning) for the knobs in these examples.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
