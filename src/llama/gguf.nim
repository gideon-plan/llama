{.experimental: "strictFuncs".}
## Pure-Nim GGUF header and KV metadata reader.
## Reads GGUF magic/version, KV pairs, and tensor info table.
## Does not read tensor weight data.

import std/[streams, tables]
import basis/code/choice
export choice

when cpuEndian == bigEndian:
  {.error: "gguf: GGUF format is little-endian; big-endian CPUs are not supported".}

const ggufMagic = "GGUF"

type
  GgufType* {.pure.} = enum
    Uint8   = 0
    Int8    = 1
    Uint16  = 2
    Int16   = 3
    Uint32  = 4
    Int32   = 5
    Float32 = 6
    Bool    = 7
    String  = 8
    Array   = 9
    Uint64  = 10
    Int64   = 11
    Float64 = 12

  GgufArray* = object
    elem_type*: GgufType
    count*:     uint64

  GgufValue* = object
    case kind*: GgufType
    of GgufType.Uint8:   u8*:  uint8
    of GgufType.Int8:    i8*:  int8
    of GgufType.Uint16:  u16*: uint16
    of GgufType.Int16:   i16*: int16
    of GgufType.Uint32:  u32*: uint32
    of GgufType.Int32:   i32*: int32
    of GgufType.Float32: f32*: float32
    of GgufType.Bool:    b*:   bool
    of GgufType.String:  s*:   string
    of GgufType.Array:   arr*: GgufArray
    of GgufType.Uint64:  u64*: uint64
    of GgufType.Int64:   i64*: int64
    of GgufType.Float64: f64*: float64

  GgufTensorMeta* = object
    name*:      string
    dims*:      seq[uint64]
    elem_type*: int32
    offset*:    uint64

  GgufMeta* = object
    version*:      uint32
    tensor_count*: uint64
    kv*:           Table[string, GgufValue]
    tensors*:      seq[GgufTensorMeta]

# =====================================================================================================================
# Stream reading primitives (little-endian; big-endian rejected at compile time above)
# =====================================================================================================================

proc readU8(s: Stream): uint8 =
  var v: uint8
  if s.readData(addr v, 1) != 1: raise newException(IOError, "gguf: truncated")
  v

proc readI8(s: Stream): int8 =
  var v: int8
  if s.readData(addr v, 1) != 1: raise newException(IOError, "gguf: truncated")
  v

proc readU16(s: Stream): uint16 =
  var v: uint16
  if s.readData(addr v, 2) != 2: raise newException(IOError, "gguf: truncated")
  v

proc readI16(s: Stream): int16 =
  var v: int16
  if s.readData(addr v, 2) != 2: raise newException(IOError, "gguf: truncated")
  v

proc readU32(s: Stream): uint32 =
  var v: uint32
  if s.readData(addr v, 4) != 4: raise newException(IOError, "gguf: truncated")
  v

proc readI32(s: Stream): int32 =
  var v: int32
  if s.readData(addr v, 4) != 4: raise newException(IOError, "gguf: truncated")
  v

proc readF32(s: Stream): float32 =
  var v: float32
  if s.readData(addr v, 4) != 4: raise newException(IOError, "gguf: truncated")
  v

proc readU64(s: Stream): uint64 =
  var v: uint64
  if s.readData(addr v, 8) != 8: raise newException(IOError, "gguf: truncated")
  v

proc readI64(s: Stream): int64 =
  var v: int64
  if s.readData(addr v, 8) != 8: raise newException(IOError, "gguf: truncated")
  v

proc readF64(s: Stream): float64 =
  var v: float64
  if s.readData(addr v, 8) != 8: raise newException(IOError, "gguf: truncated")
  v

proc readGgufStr(s: Stream): string =
  let len = readU64(s)
  if len == 0:
    return ""
  if len > 64 * 1024 * 1024:
    raise newException(IOError, "gguf: string length exceeds 64 MiB: " & $len)
  result = newString(int(len))
  if s.readData(addr result[0], int(len)) != int(len):
    raise newException(IOError, "gguf: truncated string")

proc readGgufType(s: Stream): GgufType =
  let raw = readU32(s)
  if raw > uint32(GgufType.high):
    raise newException(ValueError, "gguf: unknown value type: " & $raw)
  GgufType(int(raw))

func elemSize(t: GgufType): int =
  case t
  of GgufType.Uint8, GgufType.Int8, GgufType.Bool: 1
  of GgufType.Uint16, GgufType.Int16: 2
  of GgufType.Uint32, GgufType.Int32, GgufType.Float32: 4
  of GgufType.Uint64, GgufType.Int64, GgufType.Float64: 8
  of GgufType.String, GgufType.Array: 0

proc skipArray(s: Stream, elem_type: GgufType, count: uint64) =
  case elem_type
  of GgufType.String:
    for _ in 0'u64 ..< count:
      let len = readU64(s)
      if len > 0:
        s.setPosition(s.getPosition() + int(len))
  of GgufType.Array:
    raise newException(ValueError, "gguf: nested arrays are not supported")
  else:
    let sz = elemSize(elem_type)
    if sz <= 0:
      raise newException(ValueError, "gguf: cannot skip element type " & $elem_type)
    s.setPosition(s.getPosition() + int(count) * sz)

proc readGgufValue(s: Stream): GgufValue =
  let t = readGgufType(s)
  case t
  of GgufType.Uint8:   GgufValue(kind: GgufType.Uint8,   u8:  readU8(s))
  of GgufType.Int8:    GgufValue(kind: GgufType.Int8,    i8:  readI8(s))
  of GgufType.Uint16:  GgufValue(kind: GgufType.Uint16,  u16: readU16(s))
  of GgufType.Int16:   GgufValue(kind: GgufType.Int16,   i16: readI16(s))
  of GgufType.Uint32:  GgufValue(kind: GgufType.Uint32,  u32: readU32(s))
  of GgufType.Int32:   GgufValue(kind: GgufType.Int32,   i32: readI32(s))
  of GgufType.Float32: GgufValue(kind: GgufType.Float32, f32: readF32(s))
  of GgufType.Bool:    GgufValue(kind: GgufType.Bool,    b:   readU8(s) != 0)
  of GgufType.String:  GgufValue(kind: GgufType.String,  s:   readGgufStr(s))
  of GgufType.Array:
    let et = readGgufType(s)
    let count = readU64(s)
    skipArray(s, et, count)
    GgufValue(kind: GgufType.Array, arr: GgufArray(elem_type: et, count: count))
  of GgufType.Uint64:  GgufValue(kind: GgufType.Uint64,  u64: readU64(s))
  of GgufType.Int64:   GgufValue(kind: GgufType.Int64,   i64: readI64(s))
  of GgufType.Float64: GgufValue(kind: GgufType.Float64, f64: readF64(s))

proc readGgufTensorMeta(s: Stream): GgufTensorMeta =
  let name = readGgufStr(s)
  let n_dims = readU32(s)
  if n_dims > 4:
    raise newException(ValueError, "gguf: tensor '" & name & "' has too many dimensions: " & $n_dims)
  var dims = newSeq[uint64](int(n_dims))
  for i in 0 ..< int(n_dims):
    dims[i] = readU64(s)
  let et = readI32(s)
  let offset = readU64(s)
  GgufTensorMeta(name: name, dims: dims, elem_type: et, offset: offset)

# =====================================================================================================================
# Public API
# =====================================================================================================================

proc parseGguf*(path: string): Choice[GgufMeta] =
  ## Parse GGUF file header, KV metadata, and tensor info table.
  ## Does not load tensor weight data.
  var s: FileStream
  try:
    s = openFileStream(path, fmRead)
  except IOError:
    return bad[GgufMeta]("gguf", "cannot open: " & path)
  defer: s.close()
  try:
    var magic = newString(4)
    if s.readData(addr magic[0], 4) != 4:
      return bad[GgufMeta]("gguf", "file too short for magic: " & path)
    if magic != ggufMagic:
      return bad[GgufMeta]("gguf", "invalid magic (got '" & magic & "'): " & path)
    let version = readU32(s)
    if version < 2 or version > 3:
      return bad[GgufMeta]("gguf", "unsupported GGUF version " & $version & ": " & path)
    let tensor_count = readU64(s)
    let kv_count = readU64(s)
    if kv_count > 100_000:
      return bad[GgufMeta]("gguf", "KV count unreasonably large: " & $kv_count)
    if tensor_count > 1_000_000:
      return bad[GgufMeta]("gguf", "tensor count unreasonably large: " & $tensor_count)
    var kv = initTable[string, GgufValue]()
    for _ in 0'u64 ..< kv_count:
      let key = readGgufStr(s)
      let val = readGgufValue(s)
      kv[key] = val
    var tensors = newSeqOfCap[GgufTensorMeta](int(tensor_count))
    for _ in 0'u64 ..< tensor_count:
      tensors.add(readGgufTensorMeta(s))
    good(GgufMeta(version: version, tensor_count: tensor_count, kv: kv, tensors: tensors))
  except IOError as e:
    bad[GgufMeta]("gguf", e.msg)
  except ValueError as e:
    bad[GgufMeta]("gguf", e.msg)

func getStr*(m: GgufMeta, key: string): Choice[string] =
  ## Return a string KV value, or None if missing or wrong type.
  if key notin m.kv: return none[string]()
  let v = m.kv[key]
  if v.kind == GgufType.String: good(v.s) else: none[string]()

func getU32*(m: GgufMeta, key: string): Choice[uint32] =
  ## Return a uint32 KV value, or None if missing or wrong type.
  if key notin m.kv: return none[uint32]()
  let v = m.kv[key]
  if v.kind == GgufType.Uint32: good(v.u32) else: none[uint32]()

func getF32*(m: GgufMeta, key: string): Choice[float32] =
  ## Return a float32 KV value, or None if missing or wrong type.
  if key notin m.kv: return none[float32]()
  let v = m.kv[key]
  if v.kind == GgufType.Float32: good(v.f32) else: none[float32]()

func getU64*(m: GgufMeta, key: string): Choice[uint64] =
  ## Return a uint64 KV value, or None if missing or wrong type.
  if key notin m.kv: return none[uint64]()
  let v = m.kv[key]
  if v.kind == GgufType.Uint64: good(v.u64) else: none[uint64]()
