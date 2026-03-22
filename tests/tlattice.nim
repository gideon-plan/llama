## Unit tests for Choice/Life lattice types.

import std/unittest
import llama/lattice

suite "Result[T, E]":
  test "good construction and predicates":
    let r = Result[int, string].good(42)
    check r.is_good
    check not r.is_bad
    check r.val == 42

  test "bad construction and predicates":
    let r = Result[int, string].bad("oops")
    check r.is_bad
    check not r.is_good
    check r.err == "oops"

  test "get_or returns value on good":
    let r = Result[int, string].good(42)
    check r.get_or(0) == 42

  test "get_or returns default on bad":
    let r = Result[int, string].bad("oops")
    check r.get_or(0) == 0

  test "map transforms good value":
    let r = Result[int, string].good(21)
    let mapped = r.map(proc(v: int): int = v * 2)
    check mapped.is_good
    check mapped.val == 42

  test "map preserves bad":
    let r = Result[int, string].bad("err")
    let mapped = r.map(proc(v: int): int = v * 2)
    check mapped.is_bad
    check mapped.err == "err"

  test "flat_map chains good":
    let r = Result[int, string].good(21)
    let chained = r.flat_map(proc(v: int): Result[int, string] = Result[int, string].good(v * 2))
    check chained.is_good
    check chained.val == 42

  test "flat_map short-circuits bad":
    let r = Result[int, string].bad("err")
    let chained = r.flat_map(proc(v: int): Result[int, string] = Result[int, string].good(v * 2))
    check chained.is_bad
    check chained.err == "err"

  test "fold dispatches on good":
    let r = Result[int, string].good(42)
    let s = r.fold(
      proc(v: int): string = "got " & $v,
      proc(e: string): string = "err: " & e)
    check s == "got 42"

  test "fold dispatches on bad":
    let r = Result[int, string].bad("oops")
    let s = r.fold(
      proc(v: int): string = "got " & $v,
      proc(e: string): string = "err: " & e)
    check s == "err: oops"

suite "Option[T]":
  test "some construction":
    let o = Option[int].some(42)
    check o.is_some
    check not o.is_none
    check o.val == 42

  test "none construction":
    let o = Option[int].none()
    check o.is_none
    check not o.is_some

  test "get_or returns value on some":
    let o = Option[int].some(42)
    check o.get_or(0) == 42

  test "get_or returns default on none":
    let o = Option[int].none()
    check o.get_or(0) == 0

  test "map transforms some":
    let o = Option[int].some(21)
    let mapped = o.map(proc(v: int): int = v * 2)
    check mapped.is_some
    check mapped.val == 42

  test "map preserves none":
    let o = Option[int].none()
    let mapped = o.map(proc(v: int): int = v * 2)
    check mapped.is_none

suite "Slop[T]":
  test "clean construction":
    let s = Slop[int].clean(42)
    check s.is_clean
    check not s.is_degraded
    check s.val == 42

  test "degraded construction":
    let s = Slop[int].dirty(42, "low confidence")
    check s.is_degraded
    check not s.is_clean
    check s.degraded == 42
    check s.reason == "low confidence"

  test "map transforms clean":
    let s = Slop[int].clean(21)
    let mapped = s.map(proc(v: int): int = v * 2)
    check mapped.is_clean
    check mapped.val == 42

  test "map transforms degraded preserving reason":
    let s = Slop[int].dirty(21, "hot")
    let mapped = s.map(proc(v: int): int = v * 2)
    check mapped.is_degraded
    check mapped.degraded == 42
    check mapped.reason == "hot"

suite "Risk[T, E]":
  test "good construction":
    let r = Risk[int, string].good(42)
    check r.is_good
    check not r.is_ugly
    check not r.is_bad
    check r.val == 42

  test "ugly construction":
    let r = Risk[int, string].ugly(42, "quantized")
    check r.is_ugly
    check r.ugly_val == 42
    check r.ugly_msg == "quantized"

  test "bad construction":
    let r = Risk[int, string].bad("fatal")
    check r.is_bad
    check r.err == "fatal"

  test "get_or returns value on good":
    check Risk[int, string].good(42).get_or(0) == 42

  test "get_or returns ugly value on ugly":
    check Risk[int, string].ugly(42, "meh").get_or(0) == 42

  test "get_or returns default on bad":
    check Risk[int, string].bad("err").get_or(0) == 0

  test "map transforms all value-carrying variants":
    let g = Risk[int, string].good(21).map(proc(v: int): int = v * 2)
    check g.is_good
    check g.val == 42
    let u = Risk[int, string].ugly(21, "low").map(proc(v: int): int = v * 2)
    check u.is_ugly
    check u.ugly_val == 42
    let b = Risk[int, string].bad("err").map(proc(v: int): int = v * 2)
    check b.is_bad

  test "fold dispatches all three":
    let g = Risk[int, string].good(42)
    let gs = g.fold(
      proc(v: int): string = "good: " & $v,
      proc(v: int, msg: string): string = "ugly: " & $v & " (" & msg & ")",
      proc(e: string): string = "bad: " & e)
    check gs == "good: 42"

    let u = Risk[int, string].ugly(42, "quantized")
    let us = u.fold(
      proc(v: int): string = "good: " & $v,
      proc(v: int, msg: string): string = "ugly: " & $v & " (" & msg & ")",
      proc(e: string): string = "bad: " & e)
    check us == "ugly: 42 (quantized)"

suite "Maybe[T, E]":
  test "good construction":
    let m = Maybe[int, string].good(42)
    check m.is_good
    check m.val == 42

  test "none construction":
    let m = Maybe[int, string].none()
    check m.is_none

  test "bad construction":
    let m = Maybe[int, string].bad("err")
    check m.is_bad
    check m.err == "err"

suite "Choice[T] (full quad)":
  test "all four states":
    let g = Choice[int].good(42)
    check g.is_good
    check g.val == 42

    let u = Choice[int].ugly(42, "meh")
    check u.is_ugly
    check u.ugly_val == 42

    let b = Choice[int].bad("err")
    check b.is_bad
    check b.err == "err"

    let n = Choice[int].absent()
    check n.is_none

suite "Life lattice":
  test "Fresh can activate":
    check Fresh.can_activate
    check not Live.can_activate

  test "Live can wind down":
    check Live.can_wind_down
    check not Fresh.can_wind_down

  test "Live and Trail can finalize":
    check Live.can_finalize
    check Trail.can_finalize
    check not Fresh.can_finalize
    check not Done.can_finalize

  test "only Live is usable":
    check Live.is_usable
    check not Fresh.is_usable
    check not Trail.is_usable
    check not Done.is_usable

  test "lifecycle progression":
    var s = Fresh
    check s.can_activate
    s = Live
    check s.is_usable
    s = Trail
    check s.can_finalize
    s = Done
    check not s.is_usable
