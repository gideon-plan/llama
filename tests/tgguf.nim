{.experimental: "strictFuncs".}
## Tests for the GGUF metadata/KV reader.
## Run with: nim r --define:ggufIntrospect --path:src tests/tgguf.nim
## For real model: LLAMA_MODEL_PATH=model/Qwen3-0.6B-Q4_0.gguf nim r --define:ggufIntrospect --path:src tests/tgguf.nim

import std/[unittest, os, strutils]
import llama/gguf

# =====================================================================================================================
# Minimal GGUF binary builder for unit tests
# =====================================================================================================================

proc append[T](buf: var string, v: T) =
  let old = buf.len
  buf.setLen(old + sizeof(T))
  var v2 = v
  copyMem(addr buf[old], addr v2, sizeof(T))

proc appendStr(buf: var string, s: string) =
  buf.append(s.len.uint64)
  buf.add(s)

proc appendKvStr(buf: var string, key, val: string) =
  appendStr(buf, key)
  buf.append(uint32(8))  # GgufType.String
  appendStr(buf, val)

proc appendKvU32(buf: var string, key: string, val: uint32) =
  appendStr(buf, key)
  buf.append(uint32(4))  # GgufType.Uint32
  buf.append(val)

proc appendKvF32(buf: var string, key: string, val: float32) =
  appendStr(buf, key)
  buf.append(uint32(6))  # GgufType.Float32
  buf.append(val)

proc appendTensor(buf: var string, name: string, dims: seq[uint64], elem_type: int32, offset: uint64) =
  appendStr(buf, name)
  buf.append(uint32(dims.len))
  for d in dims:
    buf.append(d)
  buf.append(elem_type)
  buf.append(offset)

proc buildMinimalGguf(): string =
  var buf = ""
  buf.add("GGUF")                   # magic
  buf.append(uint32(3))             # version
  buf.append(uint64(1))             # tensor_count
  buf.append(uint64(4))             # kv_count
  appendKvStr(buf, "general.architecture", "llama")
  appendKvU32(buf, "llama.embedding_length", 4096)
  appendKvU32(buf, "llama.block_count", 32)
  appendKvF32(buf, "llama.rope.freq_base", 500000.0'f32)
  appendTensor(buf, "token_embd.weight", @[uint64(4096), uint64(32000)], 10'i32, 0'u64)
  buf

proc writeTmp(content: string): string =
  result = getTempDir() / "tgguf_test.gguf"
  writeFile(result, content)

# =====================================================================================================================
# Tests
# =====================================================================================================================

suite "GGUF parser - minimal binary":
  var tmp_path: string

  setup:
    tmp_path = writeTmp(buildMinimalGguf())

  teardown:
    if fileExists(tmp_path):
      removeFile(tmp_path)

  test "parseGguf returns good on valid file":
    let r = parseGguf(tmp_path)
    check r.is_good

  test "version is 3":
    let r = parseGguf(tmp_path)
    check r.is_good
    check r.val.version == 3'u32

  test "tensor_count is 1":
    let r = parseGguf(tmp_path)
    check r.is_good
    check r.val.tensor_count == 1'u64
    check r.val.tensors.len == 1

  test "getStr returns architecture":
    let r = parseGguf(tmp_path)
    check r.is_good
    let arch = r.val.getStr("general.architecture")
    check arch.is_good
    check arch.val == "llama"

  test "getU32 returns embedding length":
    let r = parseGguf(tmp_path)
    check r.is_good
    let embd = r.val.getU32("llama.embedding_length")
    check embd.is_good
    check embd.val == 4096'u32

  test "getU32 returns block count":
    let r = parseGguf(tmp_path)
    check r.is_good
    let n = r.val.getU32("llama.block_count")
    check n.is_good
    check n.val == 32'u32

  test "getF32 returns rope freq base":
    let r = parseGguf(tmp_path)
    check r.is_good
    let f = r.val.getF32("llama.rope.freq_base")
    check f.is_good
    check abs(f.val - 500000.0'f32) < 1.0'f32

  test "getStr returns none for missing key":
    let r = parseGguf(tmp_path)
    check r.is_good
    check r.val.getStr("no.such.key").is_none

  test "getU32 returns none for wrong type":
    let r = parseGguf(tmp_path)
    check r.is_good
    # general.architecture is a string, not uint32
    check r.val.getU32("general.architecture").is_none

  test "tensor info is correct":
    let r = parseGguf(tmp_path)
    check r.is_good
    let t = r.val.tensors[0]
    check t.name == "token_embd.weight"
    check t.dims == @[uint64(4096), uint64(32000)]
    check t.elem_type == 10'i32
    check t.offset == 0'u64

suite "GGUF parser - error cases":
  test "bad path returns bad":
    let r = parseGguf("/no/such/file.gguf")
    check r.is_bad

  test "wrong magic returns bad":
    let tmp = getTempDir() / "tgguf_bad_magic.gguf"
    writeFile(tmp, "NOTGGUF\x03\x00\x00\x00")
    defer: removeFile(tmp)
    let r = parseGguf(tmp)
    check r.is_bad
    check "invalid magic" in r.err.msg

  test "unsupported version returns bad":
    var buf = ""
    buf.add("GGUF")
    buf.append(uint32(99))  # unsupported version
    buf.append(uint64(0))
    buf.append(uint64(0))
    let tmp = getTempDir() / "tgguf_bad_ver.gguf"
    writeFile(tmp, buf)
    defer: removeFile(tmp)
    let r = parseGguf(tmp)
    check r.is_bad
    check "unsupported" in r.err.msg

suite "GGUF parser - real model":
  let model_path = getEnv("LLAMA_MODEL_PATH", "")

  test "parse real model metadata":
    if model_path.len == 0:
      skip()
    else:
      let r = parseGguf(model_path)
      check r.is_good
      let arch = r.val.getStr("general.architecture")
      check arch.is_good
      check arch.val.len > 0
      check r.val.tensor_count > 0
      check r.val.tensors.len > 0
