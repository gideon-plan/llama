## Integration tests for llama.cpp API.
## Requires LLAMA_MODEL_PATH env var pointing to a GGUF model file.
## Example: LLAMA_MODEL_PATH=~/.cache/models/qwen3-0.6b-q4_0.gguf nim r tests/tapi.nim

import std/[unittest, os, strutils]
import llama

const model_path = getEnv("LLAMA_MODEL_PATH", "")

when model_path.len == 0:
  # runtime check
  discard
else:
  discard

suite "Model lifecycle":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "load invalid path returns bad":
    let r = load_model("/nonexistent/model.gguf")
    check r.is_bad
    check "failed to load" in r.err.msg

  test "load model from env path":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      let r = load_model(path)
      check r.is_good
      var m = r.val
      check m.state == Live
      # Accessors
      check m.n_embd.is_some
      check m.n_layer.is_some
      check m.n_params.is_some
      check m.description.is_some
      # Close
      m.close()
      check m.state == Done
      # Accessors after close return None
      check m.n_embd.is_none

suite "Context lifecycle":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "context on dead model returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      let cr = m.init_context()
      check cr.is_bad

  test "context creation succeeds":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let cr = m.init_context(n_ctx = 512, n_batch = 256)
      check cr.is_good
      var ctx = cr.val
      check ctx.state == Live
      check ctx.n_ctx > 0
      ctx.close()
      check ctx.state == Done
      m.close()

suite "Tokenization":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "tokenize and detokenize round-trip":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let tr = m.tokenize("Hello, world!")
      check tr.is_good
      let tokens = tr.val
      check tokens.len > 0
      let dr = m.detokenize(tokens)
      check dr.is_good
      check "Hello" in dr.val
      m.close()

  test "tokenize empty string":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      let tr = m.tokenize("")
      check tr.is_good
      # May contain BOS token only
      m.close()

  test "tokenize on dead model returns bad":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      m.close()
      let tr = m.tokenize("hello")
      check tr.is_bad

suite "Decode":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "decode prompt tokens":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 512).val
      let tokens = m.tokenize("The capital of France is").val
      let dr = ctx.decode(tokens)
      check dr.is_good
      ctx.close()
      m.close()

suite "Sampling":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "sample after decode":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 512).val
      let tokens = m.tokenize("Once upon a time").val
      discard ctx.decode(tokens)
      let sr = init_sampler(temperature = 0.0)
      check sr.is_good
      var sampler = sr.val
      let s = sampler.sample(ctx)
      check s.is_clean
      check s.val != LLAMA_TOKEN_NULL
      sampler.close()
      ctx.close()
      m.close()

suite "Generation iterator":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "generate tokens from prompt":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 256, n_batch = 256).val
      var sampler = init_sampler(temperature = 0.0).val
      let tokens = m.tokenize("The meaning of life is").val
      var generated: seq[LlamaToken]
      for tok in ctx.generate(sampler, tokens, max_tokens = 16):
        if tok.is_bad:
          break
        generated.add(tok.get_or(LLAMA_TOKEN_NULL))
      check generated.len > 0
      let text = m.detokenize(generated)
      check text.is_good
      sampler.close()
      ctx.close()
      m.close()

suite "Embeddings":
  setup:
    init_backend()

  teardown:
    free_backend()

  test "extract embeddings":
    let path = getEnv("LLAMA_MODEL_PATH", "")
    if path.len == 0:
      skip()
    else:
      var m = load_model(path).val
      var ctx = m.init_context(n_ctx = 512, embeddings = true).val
      let tokens = m.tokenize("Hello, world!").val
      discard ctx.decode(tokens)
      let er = ctx.get_embeddings()
      # May succeed or fail depending on model type
      if er.is_good:
        let embd = er.val
        check embd.len == m.n_embd.get_or(0)
      ctx.close()
      m.close()
