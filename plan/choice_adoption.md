# Choice/Life Adoption Plan: llama

## Summary

- **lattice.nim**: full pilot lattice (Result, Option, Slop, Risk, Maybe, Choice, Life)
- **Error type**: `LlamaError` defined in `api.nim` -- stays in place
- **Files to modify**: 3 (api.nim, tests/tlattice.nim, tests/tstress.nim)
- **Call sites**: 28 Result + 13 Option + 4 Slop + 14 Risk = ~60 total
- **Life**: Yes. Model, Context, SamplerChain track state (Live/Done)

## lattice.nim disposition

Delete entirely. All 6 types (Result, Option, Slop, Risk, Maybe, Choice) replaced by single `Choice[T]` from `basis/code/choice`. Life replaced by `basis/code/choice.Life`.

## Type mapping

| Old type | New type | Old constructor | New constructor |
|----------|----------|----------------|----------------|
| `Result[T, LlamaError]` | `Choice[T]` | `.good(v)` | `good(v)` |
| `Result[T, LlamaError]` | `Choice[T]` | `.bad(LlamaError(msg: "x"))` | `bad[T]("llama", "x")` |
| `Option[T]` | `Choice[T]` | `.some(v)` | `good(v)` |
| `Option[T]` | `Choice[T]` | `.none()` | `none[T]()` |
| `Slop[T]` | `Choice[T]` | `.clean(v)` | `good(v)` |
| `Slop[T]` | `Choice[T]` | `.dirty(v, reason)` | `ugly(v, reason)` |
| `Risk[T, E]` | `Choice[T]` | `.good(v)` | `good(v)` |
| `Risk[T, E]` | `Choice[T]` | `.ugly(v, msg)` | `ugly(v, msg)` |
| `Risk[T, E]` | `Choice[T]` | `.bad(LlamaError(msg: "x"))` | `bad[T]("llama", "x")` |

## Predicate mapping

| Old | New |
|-----|-----|
| `is_some` | `is_good` |
| `is_none` (Option) | `is_none` |
| `is_clean` | `is_good` |
| `is_degraded` | `is_ugly` |
| `is_ugly` (Risk) | `is_ugly` |
| `is_usable` (Life) | `== Life.Live` |

## Life adoption

Model, Context, SamplerChain already use a 4-state Life. Replace with canonical 7-state Life from basis:

| Old | New |
|-----|-----|
| `Fresh` | `Life.Fresh` (unused in practice) |
| `Live` | `Life.Live` |
| `Trail` | `Life.Trail` (unused) |
| `Done` | `Life.Done` |
| `is_usable` | `== Life.Live` |
| `can_activate` | `valid_transition(s, LifeOp.Keep)` |

## api.nim specific changes

- `generate` iterator: yields `Choice[LlamaToken]` instead of `Risk[LlamaToken, LlamaError]`
- `sample`: returns `Choice[LlamaToken]` instead of `Slop[LlamaToken]`
- `decode`: returns `Choice[DecodeStatus]` instead of `Risk[DecodeStatus, LlamaError]`
  - KV slot miss: `ugly(dsNoKvSlot, "no KV slot", Trust.Probable)`
  - Fatal: `bad[DecodeStatus]("llama", "fatal decode error")`
- Model accessors: return `Choice[T]` instead of `Option[T]`
  - Live: `good(value)`, Not live: `none[T]()`

## Test disposition

- `tests/tlattice.nim`: rewrite to test `basis/code/choice` via `llama/api` re-export
- `tests/tstress.nim`: update constructor calls
