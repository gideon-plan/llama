{.experimental: "strictFuncs".}
## llama.cpp Nim binding with Choice/Life lattice types.

import llama/api
export api

when defined(ggufIntrospect):
  import llama/gguf
  export gguf
