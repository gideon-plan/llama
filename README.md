# llama

Nim binding for llama.cpp with Choice/Life lattice types.

## llama.cpp Version

| Field | Value |
|-------|-------|
| Tag | b8416 |
| Date | 2026-03-18 |
| Repo | https://github.com/ggml-org/llama.cpp |

## Build

Requires `libllama.so` (or `.dylib`/`.dll`) built from the pinned tag.

```sh
git clone --branch b8416 --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF
cmake --build build --config Release
```

Set `LD_LIBRARY_PATH` to include the build output directory containing `libllama.so`.
