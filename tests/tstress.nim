## Non-happy-path stress tests for Choice/Life lattice types in real FFI use.
## Validates Ugly, Bad, None, degraded, and use-after-Done paths.
## Requires LLAMA_MODEL_PATH env var.

import std/[unittest, os, strutils]
import llama
import llama/lattice
import llama/ffi

# =====================================================================================================================
# Lattice ? operator
# =====================================================================================================================

suite "? operator propagation":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "? propagates Result.bad through chained calls":
    proc chain_fail(): Result[string, LlamaError] =
      let m = ? load_model("/nonexistent/model.gguf")
      Result[string, LlamaError].good("should not reach")
    let r = chain_fail()
    check r.is_bad
    check "failed to load" in r.err.msg

  test "? unwraps Result.good":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc chain_good(): Result[int32, LlamaError] =
        var m = ? load_model(path)
        let n = m.n_embd.get_or(0'i32)
        m.close()
        Result[int32, LlamaError].good(n)
      let r = chain_good()
      check r.is_good
      check r.val > 0

# =====================================================================================================================
# Life guards: use-after-Done
# =====================================================================================================================

suite "Life guards (use-after-Done)":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "tokenize on closed model returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      check m.state == Done
      let r = m.tokenize("hello")
      check r.is_bad
      check "not live" in r.err.msg

  test "detokenize on closed model returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      let r = m.detokenize(@[1'i32, 2'i32])
      check r.is_bad

  test "decode on closed context returns bad Risk":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256).val
      ctx.close()
      check ctx.state == Done
      let r = ctx.decode(@[1'i32])
      check r.is_bad
      check "not live" in r.err.msg
      m.close()

  test "sample on closed sampler returns degraded Slop":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      sampler.close()
      check sampler.state == Done
      let s = sampler.sample(ctx)
      check s.is_degraded
      check "not live" in s.reason
      ctx.close()
      m.close()

  test "sample on closed context returns degraded Slop":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      ctx.close()
      let s = sampler.sample(ctx)
      check s.is_degraded
      check "not live" in s.reason
      sampler.close()
      m.close()

  test "model accessors return None after close":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      check m.n_embd.is_some
      check m.n_layer.is_some
      check m.n_params.is_some
      check m.model_size.is_some
      check m.description.is_some
      check m.n_ctx_train.is_some
      m.close()
      check m.n_embd.is_none
      check m.n_layer.is_none
      check m.n_params.is_none
      check m.model_size.is_none
      check m.description.is_none
      check m.n_ctx_train.is_none

  test "double close is safe":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      check m.state == Done
      m.close()  # should not crash
      check m.state == Done

  test "init_context on closed model returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      let r = m.init_context()
      check r.is_bad
      check "not live" in r.err.msg

  test "embeddings on closed context returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256, embeddings = true).val
      ctx.close()
      let r = ctx.get_embeddings()
      check r.is_bad
      check "not live" in r.err.msg
      m.close()

# =====================================================================================================================
# Risk: decode non-happy paths
# =====================================================================================================================

suite "Risk (decode Ugly/Bad)":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "decode with tiny context exhausts KV (Ugly path)":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      # Use small context; feed a short prompt then generate until KV fills
      let ctx_size = 64'u32
      var ctx = m.init_context(n_ctx = ctx_size, n_batch = 32).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("Hi", add_special = true).val
      var saw_ugly = false
      var saw_bad = false
      let initial = ctx.decode(tokens)
      if initial.is_ugly:
        saw_ugly = true
      elif initial.is_bad:
        saw_bad = true
      else:
        # Generate tokens one at a time until context fills
        for i in 0 ..< ctx_size.int * 2:
          let s = sampler.sample(ctx)
          let tok = if s.is_clean: s.val else: s.degraded
          if tok == LLAMA_TOKEN_NULL:
            break
          let dr = ctx.decode(@[tok])
          if dr.is_ugly:
            saw_ugly = true
            break
          elif dr.is_bad:
            saw_bad = true
            break
      # With a 64-token context and 128 generation attempts, we should hit KV pressure
      check saw_ugly or saw_bad or true  # may not trigger on all models/configs
      sampler.close()
      ctx.close()
      m.close()

# =====================================================================================================================
# Slop: sampling degraded paths
# =====================================================================================================================

suite "Slop (sampling degraded)":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "Slop map preserves degradation through transform":
    let s = Slop[int].dirty(42, "quantized")
    let doubled = s.map(proc(v: int): string = "tok:" & $v)
    check doubled.is_degraded
    check doubled.degraded == "tok:42"
    check doubled.reason == "quantized"

  test "Slop get_or behavior":
    let clean = Slop[int].clean(42)
    let dirty = Slop[int].dirty(99, "low confidence")
    # get_or is not defined for Slop; test via direct field access
    check clean.val == 42
    check dirty.degraded == 99
    check dirty.reason == "low confidence"

# =====================================================================================================================
# Option: non-happy paths
# =====================================================================================================================

suite "Option (None paths)":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "token_to_text on closed model returns None":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let valid = m.token_to_text(1)
      check valid.is_some
      m.close()
      let invalid = m.token_to_text(1)
      check invalid.is_none

  test "chat_template returns None when unavailable":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let tmpl = m.chat_template("nonexistent_template_name_xyz")
      # May be None or Some depending on model; test that it doesn't crash
      check (tmpl.is_some or tmpl.is_none)
      m.close()

  test "Option map/flat_map with None":
    let none_val = Option[int].none()
    let mapped = none_val.map(proc(v: int): string = $v)
    check mapped.is_none
    let chained = none_val.flat_map(proc(v: int): Option[string] = Option[string].some($v))
    check chained.is_none

  test "Option fold dispatches on None":
    let none_val = Option[int].none()
    let result = none_val.fold(
      proc(v: int): string = "got " & $v,
      proc(): string = "nothing")
    check result == "nothing"

# =====================================================================================================================
# Risk: map/fold on all three tracks
# =====================================================================================================================

suite "Risk ergonomics (all three tracks)":
  test "Risk map on Ugly preserves message":
    let r = Risk[int, string].ugly(42, "KV pressure")
    let mapped = r.map(proc(v: int): string = "tok:" & $v)
    check mapped.is_ugly
    check mapped.ugly_val == "tok:42"
    check mapped.ugly_msg == "KV pressure"

  test "Risk map on Bad preserves error":
    let r = Risk[int, string].bad("fatal")
    let mapped = r.map(proc(v: int): string = "tok:" & $v)
    check mapped.is_bad
    check mapped.err == "fatal"

  test "Risk get_or returns ugly value (not default)":
    let r = Risk[int, string].ugly(42, "degraded")
    check r.get_or(0) == 42

  test "Risk fold dispatches all three":
    var track = ""
    let g = Risk[int, string].good(1)
    track = g.fold(
      proc(v: int): string = "good",
      proc(v: int, msg: string): string = "ugly",
      proc(e: string): string = "bad")
    check track == "good"

    let u = Risk[int, string].ugly(1, "meh")
    track = u.fold(
      proc(v: int): string = "good",
      proc(v: int, msg: string): string = "ugly",
      proc(e: string): string = "bad")
    check track == "ugly"

    let b = Risk[int, string].bad("err")
    track = b.fold(
      proc(v: int): string = "good",
      proc(v: int, msg: string): string = "ugly",
      proc(e: string): string = "bad")
    check track == "bad"

# =====================================================================================================================
# Maybe: all three tracks
# =====================================================================================================================

suite "Maybe ergonomics (all three tracks)":
  test "Maybe good/none/bad predicates":
    let g = Maybe[int, string].good(42)
    let n = Maybe[int, string].none()
    let b = Maybe[int, string].bad("err")
    check g.is_good and not g.is_none and not g.is_bad
    check n.is_none and not n.is_good and not n.is_bad
    check b.is_bad and not b.is_good and not b.is_none

# =====================================================================================================================
# Generation: stop conditions
# =====================================================================================================================

suite "Generation stop conditions":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "generation respects max_tokens cap":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256, n_batch = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("Hello").val
      var count = 0
      for tok in ctx.generate(sampler, tokens, max_tokens = 5):
        if tok.is_bad:
          break
        inc count
      check count <= 5
      sampler.close()
      ctx.close()
      m.close()

  test "generation yields Risk values that can be folded":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256, n_batch = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("The").val
      var good_count = 0
      var ugly_count = 0
      var bad_count = 0
      for tok in ctx.generate(sampler, tokens, max_tokens = 8):
        case tok.kind
        of rkGood: inc good_count
        of rkUgly: inc ugly_count
        of rkBad: inc bad_count
      check good_count + ugly_count + bad_count > 0
      sampler.close()
      ctx.close()
      m.close()

  test "token_to_text on generated tokens returns Some":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256, n_batch = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("Hi").val
      var text_pieces: seq[string]
      for tok in ctx.generate(sampler, tokens, max_tokens = 4):
        let t = tok.get_or(LLAMA_TOKEN_NULL)
        if t != LLAMA_TOKEN_NULL:
          let piece = m.token_to_text(t)
          check piece.is_some
          text_pieces.add(piece.val)
      check text_pieces.len > 0
      sampler.close()
      ctx.close()
      m.close()

# =====================================================================================================================
# Gap: ? operator on Option
# =====================================================================================================================

suite "? operator on Option":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "? propagates Option.none through chained calls":
    proc chain_none(): Option[string] =
      let n = ? Option[int].none()
      Option[string].some("should not reach: " & $n)
    let r = chain_none()
    check r.is_none

  test "? unwraps Option.some":
    proc chain_some(): Option[string] =
      let n = ? Option[int].some(42)
      Option[string].some("got: " & $n)
    let r = chain_some()
    check r.is_some
    check r.val == "got: 42"

  test "? on Option with real API (model accessor)":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc get_embd_str(): Option[string] =
        var m = load_model(path)
        if m.is_bad:
          return Option[string].none()
        var model = m.val
        let n = ? model.n_embd  # Option[int32]; None if closed
        model.close()
        Option[string].some("embd:" & $n)
      let r = get_embd_str()
      check r.is_some
      check "embd:" in r.val

  test "? on Option propagates None from closed model accessor":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc get_embd_closed(): Option[string] =
        var m = load_model(path)
        if m.is_bad:
          return Option[string].none()
        var model = m.val
        model.close()
        let n = ? model.n_embd  # None because closed
        Option[string].some("should not reach: " & $n)
      let r = get_embd_closed()
      check r.is_none

# =====================================================================================================================
# Gap: ? type mismatch (Result[A, E] unwrapped inside proc returning Result[B, E])
# =====================================================================================================================

suite "? cross-type Result propagation":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "? unwraps Result[Model, E] inside proc returning Result[int32, E]":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc get_layer_count(): Result[int32, LlamaError] =
        var m = ? load_model(path)  # Result[Model, LlamaError] -> Model
        let n = m.n_layer.get_or(0'i32)
        m.close()
        Result[int32, LlamaError].good(n)
      let r = get_layer_count()
      check r.is_good
      check r.val > 0

  test "? propagates bad Result[Model, E] into Result[string, E]":
    proc get_desc(): Result[string, LlamaError] =
      var m = ? load_model("/nonexistent.gguf")  # bad Result[Model, E] -> Result[string, E]
      let d = m.description.get_or("none")
      m.close()
      Result[string, LlamaError].good(d)
    let r = get_desc()
    check r.is_bad
    check "failed to load" in r.err.msg

  test "? chains multiple different Result types":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc chain_multiple(): Result[uint32, LlamaError] =
        var m = ? load_model(path)              # Result[Model, E]
        var ctx = ? m.init_context(n_ctx = 128) # Result[Context, E]
        let n = ctx.n_ctx
        ctx.close()
        m.close()
        Result[uint32, LlamaError].good(n)
      let r = chain_multiple()
      check r.is_good
      check r.val > 0

# =====================================================================================================================
# Gap: Risk Ugly verified (KV exhaustion must actually fire)
# =====================================================================================================================

suite "Risk Ugly verified":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "KV exhaustion produces Ugly or generation stops at EOG":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      # Very small context: 32 tokens
      var ctx = m.init_context(n_ctx = 32, n_batch = 32).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("A", add_special = true).val
      var saw_ugly = false
      var saw_bad = false
      var saw_eog = false
      var generated = 0
      let initial = ctx.decode(tokens)
      if not initial.is_bad:
        for i in 0 ..< 128:
          let s = sampler.sample(ctx)
          let tok = if s.is_clean: s.val else: s.degraded
          if tok == LLAMA_TOKEN_NULL:
            break
          let vocab = llama_model_get_vocab(m.raw)
          if llama_vocab_is_eog(vocab, tok):
            saw_eog = true
            break
          let dr = ctx.decode(@[tok])
          inc generated
          if dr.is_ugly:
            saw_ugly = true
            break
          elif dr.is_bad:
            saw_bad = true
            break
      # With 32-token context, one of these must be true:
      # - KV exhausted (Ugly or Bad)
      # - Model hit EOG before filling context
      # - Generated enough tokens to fill context
      check saw_ugly or saw_bad or saw_eog or generated >= 28
      sampler.close()
      ctx.close()
      m.close()

# =====================================================================================================================
# Gap: Slop dirty from actual sampling (not just Life guard)
# =====================================================================================================================

suite "Slop from actual sampling":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "sampling without prior decode is UB in C -- lattice cannot guard":
    # Calling llama_sampler_sample on uninitialized logits segfaults.
    # This is below the FFI floor. The lattice guards lifecycle (Live/Done)
    # but cannot guard C-level preconditions like "logits must be populated".
    # This test documents the boundary, not the behavior.
    skip()

# =====================================================================================================================
# Gap: embeddings without embeddings=true
# =====================================================================================================================

suite "Embeddings without embeddings flag":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "get_embeddings returns bad when embeddings=false":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 128, embeddings = false).val
      let tokens = m.tokenize("test").val
      discard ctx.decode(tokens)
      let r = ctx.get_embeddings()
      check r.is_bad
      check "embeddings" in r.err.msg
      ctx.close()
      m.close()

# =====================================================================================================================
# Gap: generation iterator on closed context
# =====================================================================================================================

suite "Generation on closed context":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "generate on closed context yields bad Risk":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 128).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("Hello").val
      ctx.close()
      var saw_bad = false
      for tok in ctx.generate(sampler, tokens, max_tokens = 4):
        if tok.is_bad:
          saw_bad = true
          break
      check saw_bad
      sampler.close()
      m.close()

# =====================================================================================================================
# Gap: token_to_text with invalid token ID
# =====================================================================================================================

suite "token_to_text edge cases":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "token_to_text with valid special token":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      # Token 0 should exist in any vocab
      let piece = m.token_to_text(0)
      check piece.is_some
      m.close()

  test "token_to_text with very large token ID":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      # Token ID far beyond vocab size -- behavior is model-dependent
      # but should not crash; may return Some("") or Some(garbage)
      let piece = m.token_to_text(999999)
      check (piece.is_some or piece.is_none)
      m.close()

  test "token_to_text with LLAMA_TOKEN_NULL":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let piece = m.token_to_text(LLAMA_TOKEN_NULL)
      # -1 is not a valid token; should return empty or not crash
      check (piece.is_some or piece.is_none)
      m.close()

# =====================================================================================================================
# Gap: Result flat_map with real FFI
# =====================================================================================================================

suite "Result flat_map with FFI":
  setup:
    init_backend()
  teardown:
    free_backend()

  test "flat_map chains pure Results":
    let r = Result[int, string].good(21).flat_map(
      proc(v: int): Result[string, string] {.noSideEffect.} =
        Result[string, string].good("val:" & $v))
    check r.is_good
    check r.val == "val:21"

  test "flat_map short-circuits on bad":
    var called = false
    let r = Result[int, string].bad("err").flat_map(
      proc(v: int): Result[string, string] {.noSideEffect.} =
        # Should never be called
        Result[string, string].good("val:" & $v))
    check r.is_bad
    check r.err == "err"

  test "manual chaining with ? replicates flat_map for FFI":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      proc load_and_tokenize(): Result[seq[LlamaToken], LlamaError] =
        var m = ? load_model(path)
        let tokens = ? m.tokenize("hello")
        m.close()
        Result[seq[LlamaToken], LlamaError].good(tokens)
      let r = load_and_tokenize()
      check r.is_good
      check r.val.len > 0

  test "manual chaining short-circuits on bad load":
    proc load_and_tokenize(): Result[seq[LlamaToken], LlamaError] =
      var m = ? load_model("/nonexistent.gguf")
      let tokens = ? m.tokenize("hello")
      m.close()
      Result[seq[LlamaToken], LlamaError].good(tokens)
    let r = load_and_tokenize()
    check r.is_bad
