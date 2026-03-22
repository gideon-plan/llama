## Choice/Life lattice types for the llama.cpp binding.
## Local definitions; extractable to basis once ergonomics are validated.

{.experimental: "strict_funcs".}

# =====================================================================================================================
# Choice lattice
# =====================================================================================================================

type
  ChoiceKind* = enum
    ckNone  ## Absent
    ckGood  ## Correct / expected
    ckUgly  ## Degraded / low confidence
    ckBad   ## Failed / error

  Choice*[T] = object
    case kind*: ChoiceKind
    of ckNone:
      discard
    of ckGood:
      val*: T
    of ckUgly:
      ugly_val*: T
      ugly_msg*: string
    of ckBad:
      err*: string

# =====================================================================================================================
# Named composites
# =====================================================================================================================

type
  Result*[T, E] = object
    ## Good | Bad. Fallible operations.
    case ok*: bool
    of true:
      val*: T
    of false:
      err*: E

  Option*[T] = object
    ## Good | None. Optional values.
    case has*: bool
    of true:
      val*: T
    of false:
      discard

  Slop*[T] = object
    ## Good | Ugly. Degraded confidence.
    case good*: bool
    of true:
      val*: T
    of false:
      degraded*: T
      reason*: string

  RiskKind* = enum
    rkGood
    rkUgly
    rkBad

  Risk*[T, E] = object
    ## Good | Ugly | Bad. Fallible + degraded.
    case kind*: RiskKind
    of rkGood:
      val*: T
    of rkUgly:
      ugly_val*: T
      ugly_msg*: string
    of rkBad:
      err*: E

  MaybeKind* = enum
    mkGood
    mkNone
    mkBad

  Maybe*[T, E] = object
    ## Good | None | Bad. Optional + fallible.
    case kind*: MaybeKind
    of mkGood:
      val*: T
    of mkNone:
      discard
    of mkBad:
      err*: E

# =====================================================================================================================
# Life lattice
# =====================================================================================================================

type
  Life* = enum
    Fresh ## Allocated, not yet active
    Live  ## Active and usable
    Trail ## Winding down / cleanup in progress
    Done  ## Finalized, no longer usable

# =====================================================================================================================
# Result constructors
# =====================================================================================================================

func good*[T, E](R: typedesc[Result[T, E]], val: T): Result[T, E] =
  Result[T, E](ok: true, val: val)

func bad*[T, E](R: typedesc[Result[T, E]], err: E): Result[T, E] =
  Result[T, E](ok: false, err: err)

# =====================================================================================================================
# Option constructors
# =====================================================================================================================

func some*[T](O: typedesc[Option[T]], val: T): Option[T] =
  Option[T](has: true, val: val)

func none*[T](O: typedesc[Option[T]]): Option[T] =
  Option[T](has: false)

# =====================================================================================================================
# Slop constructors
# =====================================================================================================================

func clean*[T](S: typedesc[Slop[T]], val: T): Slop[T] =
  Slop[T](good: true, val: val)

func dirty*[T](S: typedesc[Slop[T]], val: T, reason: string): Slop[T] =
  Slop[T](good: false, degraded: val, reason: reason)

# =====================================================================================================================
# Risk constructors
# =====================================================================================================================

func good*[T, E](R: typedesc[Risk[T, E]], val: T): Risk[T, E] =
  Risk[T, E](kind: rkGood, val: val)

func ugly*[T, E](R: typedesc[Risk[T, E]], val: T, msg: string): Risk[T, E] =
  Risk[T, E](kind: rkUgly, ugly_val: val, ugly_msg: msg)

func bad*[T, E](R: typedesc[Risk[T, E]], err: E): Risk[T, E] =
  Risk[T, E](kind: rkBad, err: err)

# =====================================================================================================================
# Maybe constructors
# =====================================================================================================================

func good*[T, E](M: typedesc[Maybe[T, E]], val: T): Maybe[T, E] =
  Maybe[T, E](kind: mkGood, val: val)

func none*[T, E](M: typedesc[Maybe[T, E]]): Maybe[T, E] =
  Maybe[T, E](kind: mkNone)

func bad*[T, E](M: typedesc[Maybe[T, E]], err: E): Maybe[T, E] =
  Maybe[T, E](kind: mkBad, err: err)

# =====================================================================================================================
# Choice constructors
# =====================================================================================================================

func good*[T](C: typedesc[Choice[T]], val: T): Choice[T] =
  Choice[T](kind: ckGood, val: val)

func ugly*[T](C: typedesc[Choice[T]], val: T, msg: string): Choice[T] =
  Choice[T](kind: ckUgly, ugly_val: val, ugly_msg: msg)

func bad*[T](C: typedesc[Choice[T]], err: string): Choice[T] =
  Choice[T](kind: ckBad, err: err)

func absent*[T](C: typedesc[Choice[T]]): Choice[T] =
  Choice[T](kind: ckNone)

# =====================================================================================================================
# Predicates
# =====================================================================================================================

func is_good*[T, E](r: Result[T, E]): bool = r.ok
func is_bad*[T, E](r: Result[T, E]): bool = not r.ok

func is_some*[T](o: Option[T]): bool = o.has
func is_none*[T](o: Option[T]): bool = not o.has

func is_clean*[T](s: Slop[T]): bool = s.good
func is_degraded*[T](s: Slop[T]): bool = not s.good

func is_good*[T, E](r: Risk[T, E]): bool = r.kind == rkGood
func is_ugly*[T, E](r: Risk[T, E]): bool = r.kind == rkUgly
func is_bad*[T, E](r: Risk[T, E]): bool = r.kind == rkBad

func is_good*[T, E](m: Maybe[T, E]): bool = m.kind == mkGood
func is_none*[T, E](m: Maybe[T, E]): bool = m.kind == mkNone
func is_bad*[T, E](m: Maybe[T, E]): bool = m.kind == mkBad

func is_good*[T](c: Choice[T]): bool = c.kind == ckGood
func is_ugly*[T](c: Choice[T]): bool = c.kind == ckUgly
func is_bad*[T](c: Choice[T]): bool = c.kind == ckBad
func is_none*[T](c: Choice[T]): bool = c.kind == ckNone

# =====================================================================================================================
# Unwrap (? operator)
# =====================================================================================================================

template `?`*[T, E](r: Result[T, E]): T =
  ## Unwrap Result or return early with the error.
  ## Enclosing proc must return Result[U, E] for some U.
  block:
    let tmp = r
    if not tmp.ok:
      return type(result)(ok: false, err: tmp.err)
    tmp.val

template `?`*[T](o: Option[T]): T =
  ## Unwrap Option or return early with None.
  ## Enclosing proc must return Option[U] for some U.
  block:
    let tmp = o
    if not tmp.has:
      return type(result)(has: false)
    tmp.val

# =====================================================================================================================
# Map / flat_map
# =====================================================================================================================

func map*[T, E, U](r: Result[T, E], f: proc(v: T): U {.noSideEffect.}): Result[U, E] =
  if r.ok:
    Result[U, E].good(f(r.val))
  else:
    Result[U, E].bad(r.err)

func flat_map*[T, E, U](r: Result[T, E], f: proc(v: T): Result[U, E] {.noSideEffect.}): Result[U, E] =
  if r.ok:
    f(r.val)
  else:
    Result[U, E].bad(r.err)

func map*[T, U](o: Option[T], f: proc(v: T): U {.noSideEffect.}): Option[U] =
  if o.has:
    Option[U].some(f(o.val))
  else:
    Option[U].none()

func flat_map*[T, U](o: Option[T], f: proc(v: T): Option[U] {.noSideEffect.}): Option[U] =
  if o.has:
    f(o.val)
  else:
    Option[U].none()

func map*[T, U](s: Slop[T], f: proc(v: T): U {.noSideEffect.}): Slop[U] =
  if s.good:
    Slop[U].clean(f(s.val))
  else:
    Slop[U].dirty(f(s.degraded), s.reason)

func map*[T, E, U](r: Risk[T, E], f: proc(v: T): U {.noSideEffect.}): Risk[U, E] =
  case r.kind
  of rkGood: Risk[U, E].good(f(r.val))
  of rkUgly: Risk[U, E].ugly(f(r.ugly_val), r.ugly_msg)
  of rkBad: Risk[U, E].bad(r.err)

# =====================================================================================================================
# Fold
# =====================================================================================================================

func fold*[T, E, U](r: Result[T, E], on_good: proc(v: T): U {.noSideEffect.}, on_bad: proc(e: E): U {.noSideEffect.}): U =
  if r.ok:
    on_good(r.val)
  else:
    on_bad(r.err)

func fold*[T, U](o: Option[T], on_some: proc(v: T): U {.noSideEffect.}, on_none: proc(): U {.noSideEffect.}): U =
  if o.has:
    on_some(o.val)
  else:
    on_none()

func fold*[T, E, U](r: Risk[T, E],
    on_good: proc(v: T): U {.noSideEffect.},
    on_ugly: proc(v: T, msg: string): U {.noSideEffect.},
    on_bad: proc(e: E): U {.noSideEffect.}): U =
  case r.kind
  of rkGood: on_good(r.val)
  of rkUgly: on_ugly(r.ugly_val, r.ugly_msg)
  of rkBad: on_bad(r.err)

# =====================================================================================================================
# get_or
# =====================================================================================================================

func get_or*[T, E](r: Result[T, E], default: T): T =
  if r.ok: r.val else: default

func get_or*[T](o: Option[T], default: T): T =
  if o.has: o.val else: default

func get_or*[T, E](r: Risk[T, E], default: T): T =
  case r.kind
  of rkGood: r.val
  of rkUgly: r.ugly_val
  of rkBad: default

# =====================================================================================================================
# Life transitions
# =====================================================================================================================

func can_activate*(s: Life): bool = s == Fresh
func can_wind_down*(s: Life): bool = s == Live
func can_finalize*(s: Life): bool = s == Trail or s == Live
func is_usable*(s: Life): bool = s == Live
