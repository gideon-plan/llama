## High-level llama.cpp API using Choice/Life lattice types.

{.experimental: "strict_funcs".}

import ./ffi
import basis/code/choice

export choice
export ffi.LlamaToken, ffi.LlamaPos, ffi.LlamaSeqId, ffi.LLAMA_TOKEN_NULL, ffi.LLAMA_DEFAULT_SEED

# =====================================================================================================================
# Error types
# =====================================================================================================================

type
  LlamaError* = object
    msg*: string

  DecodeStatus* {.pure.} = enum
    Success = 0
    NoKvSlot = 1
    Aborted = 2

# =====================================================================================================================
# Managed types with Life tracking
# =====================================================================================================================

type
  Model* = object
    raw*: ptr LlamaModel
    state*: Life

  Context* = object
    raw*: ptr LlamaContext
    model*: ptr Model
    state*: Life

  SamplerChain* = object
    raw*: ptr LlamaSampler
    state*: Life

# =====================================================================================================================
# Backend (module-level)
# =====================================================================================================================

proc init_backend*() =
  llama_backend_init()

proc free_backend*() =
  llama_backend_free()

# =====================================================================================================================
# Model lifecycle: Fresh -> Live -> Done
# =====================================================================================================================

proc load_model*(path: string, n_gpu_layers: int32 = 0): Choice[Model] =
  var params = llama_model_default_params()
  params.n_gpu_layers = n_gpu_layers
  let raw = llama_model_load_from_file(path.cstring, params)
  if raw == nil:
    bad[Model]("llama", "failed to load model: " & path)
  else:
    good(Model(raw: raw, state: Life.Live))

proc close*(m: var Model) =
  if m.state == Life.Live:
    llama_model_free(m.raw)
    m.raw = nil
    m.state = Life.Done

# Model accessors (only valid when Live)

func n_ctx_train*(m: Model): Choice[int32] =
  if m.state == Life.Live:
    good(llama_model_n_ctx_train(m.raw))
  else:
    none[int32]()

func n_embd*(m: Model): Choice[int32] =
  if m.state == Life.Live:
    good(llama_model_n_embd(m.raw))
  else:
    none[int32]()

func n_layer*(m: Model): Choice[int32] =
  if m.state == Life.Live:
    good(llama_model_n_layer(m.raw))
  else:
    none[int32]()

func n_params*(m: Model): Choice[uint64] =
  if m.state == Life.Live:
    good(llama_model_n_params(m.raw))
  else:
    none[uint64]()

func model_size*(m: Model): Choice[uint64] =
  if m.state == Life.Live:
    good(llama_model_size(m.raw))
  else:
    none[uint64]()

proc description*(m: Model): Choice[string] =
  if m.state != Life.Live:
    return none[string]()
  var buf = newString(256)
  let n = llama_model_desc(m.raw, buf.cstring, buf.len.csize_t)
  if n > 0:
    buf.setLen(n)
    good(buf)
  else:
    none[string]()

proc chat_template*(m: Model, name: string = ""): Choice[string] =
  if m.state != Life.Live:
    return none[string]()
  let tmpl_name = if name.len == 0: nil.cstring else: name.cstring
  let raw = llama_model_chat_template(m.raw, tmpl_name)
  if raw == nil:
    none[string]()
  else:
    good($raw)

# =====================================================================================================================
# Context lifecycle: Fresh -> Live -> Done
# =====================================================================================================================

proc init_context*(m: var Model, n_ctx: uint32 = 0, n_batch: uint32 = 512, n_threads: int32 = 4, embeddings: bool = false, n_seq_max: uint32 = 1): Choice[Context] =
  if m.state != Life.Live:
    return bad[Context]("llama", "model not live")
  var params = llama_context_default_params()
  params.n_ctx = n_ctx
  params.n_batch = n_batch
  params.n_threads = n_threads
  params.n_threads_batch = n_threads
  params.embeddings = embeddings
  params.n_seq_max = n_seq_max
  let raw = llama_init_from_model(m.raw, params)
  if raw == nil:
    bad[Context]("llama", "failed to create context")
  else:
    good(Context(raw: raw, model: addr m, state: Life.Live))

proc close*(c: var Context) =
  if c.state == Life.Live:
    llama_free(c.raw)
    c.raw = nil
    c.state = Life.Done

func n_ctx*(c: Context): uint32 =
  if c.state == Life.Live: llama_n_ctx(c.raw) else: 0

# =====================================================================================================================
# Tokenization: Choice[seq[LlamaToken]]
# =====================================================================================================================

proc tokenize*(m: Model, text: string, add_special: bool = true, parse_special: bool = false): Choice[seq[LlamaToken]] =
  if m.state != Life.Live:
    return bad[seq[LlamaToken]]("llama", "model not live")
  let vocab = llama_model_get_vocab(m.raw)
  # First pass: get required size
  let n_needed = llama_tokenize(vocab, text.cstring, text.len.int32, nil, 0, add_special, parse_special)
  if n_needed == int32.low:
    return bad[seq[LlamaToken]]("llama", "tokenization overflow")
  let capacity = if n_needed < 0: -n_needed else: n_needed
  if capacity == 0:
    return good(newSeq[LlamaToken]())
  var tokens = newSeq[LlamaToken](capacity)
  let n = llama_tokenize(vocab, text.cstring, text.len.int32, addr tokens[0], capacity, add_special, parse_special)
  if n < 0:
    bad[seq[LlamaToken]]("llama", "tokenization failed")
  else:
    tokens.setLen(n)
    good(tokens)

proc detokenize*(m: Model, tokens: seq[LlamaToken], remove_special: bool = false, unparse_special: bool = false): Choice[string] =
  if m.state != Life.Live:
    return bad[string]("llama", "model not live")
  if tokens.len == 0:
    return good("")
  let vocab = llama_model_get_vocab(m.raw)
  var buf = newString(tokens.len * 8)
  let n = llama_detokenize(vocab, unsafeAddr tokens[0], tokens.len.int32, buf.cstring, buf.len.int32, remove_special, unparse_special)
  if n < 0:
    # Need more space
    buf = newString(-n)
    let n2 = llama_detokenize(vocab, unsafeAddr tokens[0], tokens.len.int32, buf.cstring, buf.len.int32, remove_special, unparse_special)
    if n2 < 0:
      bad[string]("llama", "detokenization failed")
    else:
      buf.setLen(n2)
      good(buf)
  else:
    buf.setLen(n)
    good(buf)

# =====================================================================================================================
# Decoding: Choice[void] (can degrade: KV slot miss; can fail: fatal error)
# =====================================================================================================================

proc decode*(c: var Context, tokens: seq[LlamaToken]): Choice[DecodeStatus] =
  if c.state != Life.Live:
    return bad[DecodeStatus]("llama", "context not live")
  var toks = tokens
  let batch = llama_batch_get_one(addr toks[0], toks.len.int32)
  let rc = llama_decode(c.raw, batch)
  case rc
  of 0:
    good(DecodeStatus.Success)
  of 1:
    ugly(DecodeStatus.NoKvSlot, "no KV slot available; reduce batch size or increase context")
  of 2:
    ugly(DecodeStatus.Aborted, "decode aborted by callback")
  else:
    if rc == -1:
      bad[DecodeStatus]("llama", "invalid input batch")
    else:
      bad[DecodeStatus]("llama", "fatal decode error: " & $rc)

# =====================================================================================================================
# Sampling: Choice[LlamaToken] (degraded confidence from temperature/top-k)
# =====================================================================================================================

proc init_sampler*(temperature: float32 = 0.8, top_k: int32 = 40, top_p: float32 = 0.95, seed: uint32 = LLAMA_DEFAULT_SEED): Choice[SamplerChain] =
  let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
  if chain == nil:
    return bad[SamplerChain]("llama", "failed to create sampler chain")
  llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k))
  llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1))
  llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
  llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
  good(SamplerChain(raw: chain, state: Life.Live))

proc close*(s: var SamplerChain) =
  if s.state == Life.Live:
    llama_sampler_free(s.raw)
    s.raw = nil
    s.state = Life.Done

proc sample*(s: SamplerChain, c: Context, idx: int32 = -1): Choice[LlamaToken] =
  ## Sample next token. Returns Slop: clean if greedy/low-temp, degraded if stochastic.
  if s.state != Life.Live or c.state != Life.Live:
    return ugly(LLAMA_TOKEN_NULL, "sampler or context not live")
  let token = llama_sampler_sample(s.raw, c.raw, idx)
  if token == LLAMA_TOKEN_NULL:
    ugly(token, "null token sampled")
  else:
    good(token)

# =====================================================================================================================
# Embeddings: Choice[seq[float32]]
# =====================================================================================================================

proc get_embeddings*(c: Context, idx: int32 = 0): Choice[seq[float32]] =
  if c.state != Life.Live:
    return bad[seq[float32]]("llama", "context not live")
  let model = llama_get_model(c.raw)
  let n_embd = llama_model_n_embd(model)
  let embd_ptr = llama_get_embeddings_ith(c.raw, idx)
  if embd_ptr == nil:
    return bad[seq[float32]]("llama", "no embeddings available; ensure context was created with embeddings=true")
  var result_vec = newSeq[float32](n_embd)
  copyMem(addr result_vec[0], embd_ptr, n_embd * sizeof(float32))
  good(result_vec)

# =====================================================================================================================
# Generation iterator: yields Choice[LlamaToken]
# =====================================================================================================================

iterator generate*(c: var Context, sampler: SamplerChain, prompt_tokens: seq[LlamaToken], max_tokens: int = 256): Choice[LlamaToken] =
  ## Streaming generation. Yields one token at a time.
  ## Good: normal token. Ugly: KV pressure or null sample. Bad: fatal error.
  if c.state != Life.Live:
    yield bad[LlamaToken]("llama", "context not live")
  else:
    # Decode prompt
    let prompt_result = c.decode(prompt_tokens)
    if prompt_result.is_bad:
      yield bad[LlamaToken](prompt_result.err)
    else:
      let vocab = llama_model_get_vocab(llama_get_model(c.raw))
      var count = 0
      while count < max_tokens:
        let sampled = sampler.sample(c)
        let token = if sampled.is_good: sampled.val else: sampled.degraded
        if token == LLAMA_TOKEN_NULL:
          yield ugly(token, "null token")
          break
        if llama_vocab_is_eog(vocab, token):
          break
        if sampled.is_ugly:
          yield ugly(token, sampled.concern)
        else:
          yield good(token)
        # Decode the new token
        let decode_result = c.decode(@[token])
        if decode_result.is_bad:
          yield bad[LlamaToken](decode_result.err)
          break
        elif decode_result.is_ugly:
          yield ugly(token, decode_result.concern)
        inc count

# =====================================================================================================================
# Convenience: token to text
# =====================================================================================================================

proc token_to_text*(m: Model, token: LlamaToken, special: bool = false): Choice[string] =
  if m.state != Life.Live:
    return none[string]()
  let vocab = llama_model_get_vocab(m.raw)
  var buf = newString(64)
  let n = llama_token_to_piece(vocab, token, buf.cstring, buf.len.int32, 0, special)
  if n < 0:
    buf = newString(-n)
    let n2 = llama_token_to_piece(vocab, token, buf.cstring, buf.len.int32, 0, special)
    if n2 > 0:
      buf.setLen(n2)
      good(buf)
    else:
      none[string]()
  elif n > 0:
    buf.setLen(n)
    good(buf)
  else:
    good("")
