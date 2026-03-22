## High-level llama.cpp API using Choice/Life lattice types.

{.experimental: "strict_funcs".}

import ./ffi
import ./lattice

export lattice
export ffi.LlamaToken, ffi.LlamaPos, ffi.LlamaSeqId, ffi.LLAMA_TOKEN_NULL, ffi.LLAMA_DEFAULT_SEED

# =====================================================================================================================
# Error types
# =====================================================================================================================

type
  LlamaError* = object
    msg*: string

  DecodeStatus* = enum
    dsSuccess = 0
    dsNoKvSlot = 1
    dsAborted = 2

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

proc load_model*(path: string, n_gpu_layers: int32 = 0): Result[Model, LlamaError] =
  var params = llama_model_default_params()
  params.n_gpu_layers = n_gpu_layers
  let raw = llama_model_load_from_file(path.cstring, params)
  if raw == nil:
    Result[Model, LlamaError].bad(LlamaError(msg: "failed to load model: " & path))
  else:
    Result[Model, LlamaError].good(Model(raw: raw, state: Live))

proc close*(m: var Model) =
  if m.state.is_usable:
    llama_model_free(m.raw)
    m.raw = nil
    m.state = Done

# Model accessors (only valid when Live)

func n_ctx_train*(m: Model): Option[int32] =
  if m.state.is_usable:
    Option[int32].some(llama_model_n_ctx_train(m.raw))
  else:
    Option[int32].none()

func n_embd*(m: Model): Option[int32] =
  if m.state.is_usable:
    Option[int32].some(llama_model_n_embd(m.raw))
  else:
    Option[int32].none()

func n_layer*(m: Model): Option[int32] =
  if m.state.is_usable:
    Option[int32].some(llama_model_n_layer(m.raw))
  else:
    Option[int32].none()

func n_params*(m: Model): Option[uint64] =
  if m.state.is_usable:
    Option[uint64].some(llama_model_n_params(m.raw))
  else:
    Option[uint64].none()

func model_size*(m: Model): Option[uint64] =
  if m.state.is_usable:
    Option[uint64].some(llama_model_size(m.raw))
  else:
    Option[uint64].none()

proc description*(m: Model): Option[string] =
  if not m.state.is_usable:
    return Option[string].none()
  var buf = newString(256)
  let n = llama_model_desc(m.raw, buf.cstring, buf.len.csize_t)
  if n > 0:
    buf.setLen(n)
    Option[string].some(buf)
  else:
    Option[string].none()

proc chat_template*(m: Model, name: string = ""): Option[string] =
  if not m.state.is_usable:
    return Option[string].none()
  let tmpl_name = if name.len == 0: nil.cstring else: name.cstring
  let raw = llama_model_chat_template(m.raw, tmpl_name)
  if raw == nil:
    Option[string].none()
  else:
    Option[string].some($raw)

# =====================================================================================================================
# Context lifecycle: Fresh -> Live -> Done
# =====================================================================================================================

proc init_context*(m: var Model, n_ctx: uint32 = 0, n_batch: uint32 = 512, n_threads: int32 = 4, embeddings: bool = false, n_seq_max: uint32 = 1): Result[Context, LlamaError] =
  if not m.state.is_usable:
    return Result[Context, LlamaError].bad(LlamaError(msg: "model not live"))
  var params = llama_context_default_params()
  params.n_ctx = n_ctx
  params.n_batch = n_batch
  params.n_threads = n_threads
  params.n_threads_batch = n_threads
  params.embeddings = embeddings
  params.n_seq_max = n_seq_max
  let raw = llama_init_from_model(m.raw, params)
  if raw == nil:
    Result[Context, LlamaError].bad(LlamaError(msg: "failed to create context"))
  else:
    Result[Context, LlamaError].good(Context(raw: raw, model: addr m, state: Live))

proc close*(c: var Context) =
  if c.state.is_usable:
    llama_free(c.raw)
    c.raw = nil
    c.state = Done

func n_ctx*(c: Context): uint32 =
  if c.state.is_usable: llama_n_ctx(c.raw) else: 0

# =====================================================================================================================
# Tokenization: Result[seq[LlamaToken], LlamaError]
# =====================================================================================================================

proc tokenize*(m: Model, text: string, add_special: bool = true, parse_special: bool = false): Result[seq[LlamaToken], LlamaError] =
  if not m.state.is_usable:
    return Result[seq[LlamaToken], LlamaError].bad(LlamaError(msg: "model not live"))
  let vocab = llama_model_get_vocab(m.raw)
  # First pass: get required size
  let n_needed = llama_tokenize(vocab, text.cstring, text.len.int32, nil, 0, add_special, parse_special)
  if n_needed == int32.low:
    return Result[seq[LlamaToken], LlamaError].bad(LlamaError(msg: "tokenization overflow"))
  let capacity = if n_needed < 0: -n_needed else: n_needed
  if capacity == 0:
    return Result[seq[LlamaToken], LlamaError].good(newSeq[LlamaToken]())
  var tokens = newSeq[LlamaToken](capacity)
  let n = llama_tokenize(vocab, text.cstring, text.len.int32, addr tokens[0], capacity, add_special, parse_special)
  if n < 0:
    Result[seq[LlamaToken], LlamaError].bad(LlamaError(msg: "tokenization failed"))
  else:
    tokens.setLen(n)
    Result[seq[LlamaToken], LlamaError].good(tokens)

proc detokenize*(m: Model, tokens: seq[LlamaToken], remove_special: bool = false, unparse_special: bool = false): Result[string, LlamaError] =
  if not m.state.is_usable:
    return Result[string, LlamaError].bad(LlamaError(msg: "model not live"))
  if tokens.len == 0:
    return Result[string, LlamaError].good("")
  let vocab = llama_model_get_vocab(m.raw)
  var buf = newString(tokens.len * 8)
  let n = llama_detokenize(vocab, unsafeAddr tokens[0], tokens.len.int32, buf.cstring, buf.len.int32, remove_special, unparse_special)
  if n < 0:
    # Need more space
    buf = newString(-n)
    let n2 = llama_detokenize(vocab, unsafeAddr tokens[0], tokens.len.int32, buf.cstring, buf.len.int32, remove_special, unparse_special)
    if n2 < 0:
      Result[string, LlamaError].bad(LlamaError(msg: "detokenization failed"))
    else:
      buf.setLen(n2)
      Result[string, LlamaError].good(buf)
  else:
    buf.setLen(n)
    Result[string, LlamaError].good(buf)

# =====================================================================================================================
# Decoding: Risk[void, LlamaError] (can degrade: KV slot miss; can fail: fatal error)
# =====================================================================================================================

proc decode*(c: var Context, tokens: seq[LlamaToken]): Risk[DecodeStatus, LlamaError] =
  if not c.state.is_usable:
    return Risk[DecodeStatus, LlamaError].bad(LlamaError(msg: "context not live"))
  var toks = tokens
  let batch = llama_batch_get_one(addr toks[0], toks.len.int32)
  let rc = llama_decode(c.raw, batch)
  case rc
  of 0:
    Risk[DecodeStatus, LlamaError].good(dsSuccess)
  of 1:
    Risk[DecodeStatus, LlamaError].ugly(dsNoKvSlot, "no KV slot available; reduce batch size or increase context")
  of 2:
    Risk[DecodeStatus, LlamaError].ugly(dsAborted, "decode aborted by callback")
  else:
    if rc == -1:
      Risk[DecodeStatus, LlamaError].bad(LlamaError(msg: "invalid input batch"))
    else:
      Risk[DecodeStatus, LlamaError].bad(LlamaError(msg: "fatal decode error: " & $rc))

# =====================================================================================================================
# Sampling: Slop[LlamaToken] (degraded confidence from temperature/top-k)
# =====================================================================================================================

proc init_sampler*(temperature: float32 = 0.8, top_k: int32 = 40, top_p: float32 = 0.95, seed: uint32 = LLAMA_DEFAULT_SEED): Result[SamplerChain, LlamaError] =
  let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
  if chain == nil:
    return Result[SamplerChain, LlamaError].bad(LlamaError(msg: "failed to create sampler chain"))
  llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k))
  llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1))
  llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
  llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
  Result[SamplerChain, LlamaError].good(SamplerChain(raw: chain, state: Live))

proc close*(s: var SamplerChain) =
  if s.state.is_usable:
    llama_sampler_free(s.raw)
    s.raw = nil
    s.state = Done

proc sample*(s: SamplerChain, c: Context, idx: int32 = -1): Slop[LlamaToken] =
  ## Sample next token. Returns Slop: clean if greedy/low-temp, degraded if stochastic.
  if not s.state.is_usable or not c.state.is_usable:
    return Slop[LlamaToken].dirty(LLAMA_TOKEN_NULL, "sampler or context not live")
  let token = llama_sampler_sample(s.raw, c.raw, idx)
  if token == LLAMA_TOKEN_NULL:
    Slop[LlamaToken].dirty(token, "null token sampled")
  else:
    Slop[LlamaToken].clean(token)

# =====================================================================================================================
# Embeddings: Result[seq[float32], LlamaError]
# =====================================================================================================================

proc get_embeddings*(c: Context, idx: int32 = 0): Result[seq[float32], LlamaError] =
  if not c.state.is_usable:
    return Result[seq[float32], LlamaError].bad(LlamaError(msg: "context not live"))
  let model = llama_get_model(c.raw)
  let n_embd = llama_model_n_embd(model)
  let embd_ptr = llama_get_embeddings_ith(c.raw, idx)
  if embd_ptr == nil:
    return Result[seq[float32], LlamaError].bad(LlamaError(msg: "no embeddings available; ensure context was created with embeddings=true"))
  var result_vec = newSeq[float32](n_embd)
  copyMem(addr result_vec[0], embd_ptr, n_embd * sizeof(float32))
  Result[seq[float32], LlamaError].good(result_vec)

# =====================================================================================================================
# Generation iterator: yields Risk[LlamaToken, LlamaError]
# =====================================================================================================================

iterator generate*(c: var Context, sampler: SamplerChain, prompt_tokens: seq[LlamaToken], max_tokens: int = 256): Risk[LlamaToken, LlamaError] =
  ## Streaming generation. Yields one token at a time.
  ## Good: normal token. Ugly: KV pressure or null sample. Bad: fatal error.
  if not c.state.is_usable:
    yield Risk[LlamaToken, LlamaError].bad(LlamaError(msg: "context not live"))
  else:
    # Decode prompt
    let prompt_result = c.decode(prompt_tokens)
    if prompt_result.is_bad:
      yield Risk[LlamaToken, LlamaError].bad(prompt_result.err)
    else:
      let vocab = llama_model_get_vocab(llama_get_model(c.raw))
      var count = 0
      while count < max_tokens:
        let sampled = sampler.sample(c)
        let token = if sampled.is_clean: sampled.val else: sampled.degraded
        if token == LLAMA_TOKEN_NULL:
          yield Risk[LlamaToken, LlamaError].ugly(token, "null token")
          break
        if llama_vocab_is_eog(vocab, token):
          break
        if sampled.is_degraded:
          yield Risk[LlamaToken, LlamaError].ugly(token, sampled.reason)
        else:
          yield Risk[LlamaToken, LlamaError].good(token)
        # Decode the new token
        let decode_result = c.decode(@[token])
        if decode_result.is_bad:
          yield Risk[LlamaToken, LlamaError].bad(decode_result.err)
          break
        elif decode_result.is_ugly:
          yield Risk[LlamaToken, LlamaError].ugly(token, decode_result.ugly_msg)
        inc count

# =====================================================================================================================
# Convenience: token to text
# =====================================================================================================================

proc token_to_text*(m: Model, token: LlamaToken, special: bool = false): Option[string] =
  if not m.state.is_usable:
    return Option[string].none()
  let vocab = llama_model_get_vocab(m.raw)
  var buf = newString(64)
  let n = llama_token_to_piece(vocab, token, buf.cstring, buf.len.int32, 0, special)
  if n < 0:
    buf = newString(-n)
    let n2 = llama_token_to_piece(vocab, token, buf.cstring, buf.len.int32, 0, special)
    if n2 > 0:
      buf.setLen(n2)
      Option[string].some(buf)
    else:
      Option[string].none()
  elif n > 0:
    buf.setLen(n)
    Option[string].some(buf)
  else:
    Option[string].some("")
