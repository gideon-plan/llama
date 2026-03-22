#=======================================================================================================================
#== INIT ===============================================================================================================
#=======================================================================================================================

package_name = "llama"

mode = ScriptMode.Silent

import std/os

#=======================================================================================================================
#== SWITCH =============================================================================================================
#=======================================================================================================================

--mm:orc
--outdir:".out"
--verbosity:1
--line_dir:on

switch("nimcache", $CurDir/".nimcache")

--define:nim_preview_dot_like_ops
--define:nim_preview_float_roundtrip
--define:nim_strict_delete
--define:nim_no_get_random
--experimental:unicode_operators
--experimental:overloadable_enums
--experimental:strict_funcs

--style_check:usages

#=======================================================================================================================
#== VENDOR =============================================================================================================
#=======================================================================================================================

let vendor_dir = thisDir() / "vendor"
let vendor_lib = vendor_dir / "lib"
let vendor_inc = vendor_dir / "include"

when not defined(static_llama):
  # Vendor include must come before system /usr/local/include to avoid header conflicts
  switch("passc", "-I" & vendor_inc & " -DGGML_API= -DLLAMA_API=")
  switch("passl", "-L" & vendor_lib & " -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath," & vendor_lib)

when defined(code_coverage):
  switch("passc", "-fprofile-arcs -fprofile-generate -coverage")
  switch("passl", "-lgcov")

#=======================================================================================================================
#== TEST ===============================================================================================================
#=======================================================================================================================

task test, "Run all tests":
  exec "nim r --path:src tests/tlattice.nim"

when file_exists("nimble.paths"):
  include "nimble.paths"
# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config
