## Raw FFI bindings for llama.cpp C API (tag b8416).
## Non-deprecated, in-scope functions only.
## Out of scope: training, state save/load, quantization, server mode, GGML internals.

import segfaults

{.experimental: "strict_funcs".}

import std/os

const vendor_dir = currentSourcePath().parentDir.parentDir.parentDir / "vendor"
const vendor_lib = vendor_dir / "lib"
const vendor_inc = vendor_dir / "include"

when defined(static_llama):
  {.passL: vendor_lib / "libllama.a".}
  {.passL: vendor_lib / "libggml.a".}
  {.passL: vendor_lib / "libggml-base.a".}
  {.passL: vendor_lib / "libggml-cpu.a".}
  {.passL: "-lstdc++ -lm -lpthread -lgomp".}
  {.pragma: llama_ffi, cdecl, importc.}
  const llama_header = vendor_inc / "llama_types.h"
else:
  const llama_header = vendor_inc / "llama.h"
  const libllama = vendor_lib / "libllama.so"
  {.pragma: llama_ffi, cdecl, dynlib: libllama, importc.}

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  LlamaVocab* {.importc: "struct llama_vocab".} = object
  LlamaModel* {.importc: "struct llama_model".} = object
  LlamaContext* {.importc: "struct llama_context".} = object
  LlamaSampler* {.importc: "struct llama_sampler".} = object
  LlamaAdapterLora* {.importc: "struct llama_adapter_lora".} = object

  LlamaMemoryT* = pointer

  LlamaPos* = int32
  LlamaToken* = int32
  LlamaSeqId* = int32

# =====================================================================================================================
# Enums
# =====================================================================================================================

type
  LlamaVocabKind* {.pure, size: sizeof(cint).} = enum
    None = 0
    Spm = 1
    Bpe = 2
    Wpm = 3
    Ugm = 4
    Rwkv = 5
    Plamo2 = 6

  LlamaTokenKind* {.pure, size: sizeof(cint).} = enum
    Undefined = 0
    Normal = 1
    Unknown = 2
    Control = 3
    UserDefined = 4
    Unused = 5
    Byte = 6

  LlamaTokenAttr* {.pure, size: sizeof(cint).} = enum
    Undefined = 0
    Unknown = 1
    Unused = 2
    Normal = 4
    Control = 8
    UserDefined = 16
    Byte = 32
    Normalized = 64
    Lstrip = 128
    Rstrip = 256
    SingleWord = 512

  LlamaPoolingKind* {.pure, size: sizeof(cint).} = enum
    Unspecified = -1
    None = 0
    Mean = 1
    Cls = 2
    Last = 3
    Rank = 4

  LlamaAttentionKind* {.pure, size: sizeof(cint).} = enum
    Unspecified = -1
    Causal = 0
    NonCausal = 1

  LlamaFlashAttnKind* {.pure, size: sizeof(cint).} = enum
    Auto = -1
    Disabled = 0
    Enabled = 1

  LlamaSplitMode* {.pure, size: sizeof(cint).} = enum
    None = 0
    Layer = 1
    Row = 2

  LlamaRopeScalingKind* {.pure, size: sizeof(cint).} = enum
    Unspecified = -1
    None = 0
    Linear = 1
    Yarn = 2
    Longrope = 3

  LlamaParamsFitStatus* {.pure, size: sizeof(cint).} = enum
    Success = 0
    Failure = 1
    Error = 2

# =====================================================================================================================
# Structs
# =====================================================================================================================

type
  LlamaTokenData* {.importc: "llama_token_data", header: llama_header.} = object
    id*: LlamaToken
    logit*: cfloat
    p*: cfloat

  LlamaTokenDataArray* {.importc: "llama_token_data_array", header: llama_header.} = object
    data*: ptr LlamaTokenData
    size*: csize_t
    selected*: int64
    sorted*: bool

  LlamaBatch* {.importc: "llama_batch", header: llama_header.} = object
    n_tokens*: int32
    token*: ptr LlamaToken
    embd*: ptr cfloat
    pos*: ptr LlamaPos
    n_seq_id*: ptr int32
    seq_id*: ptr ptr LlamaSeqId
    logits*: ptr int8

  LlamaChatMessage* {.importc: "llama_chat_message", header: llama_header.} = object
    role*: cstring
    content*: cstring

  LlamaLogitBias* {.importc: "llama_logit_bias", header: llama_header.} = object
    token*: LlamaToken
    bias*: cfloat

  LlamaSamplerChainParams* {.importc: "llama_sampler_chain_params", header: llama_header.} = object
    no_perf*: bool

  LlamaModelParams* {.importc: "struct llama_model_params", header: llama_header.} = object
    devices*: pointer
    tensor_buft_overrides*: pointer
    n_gpu_layers*: int32
    split_mode*: LlamaSplitMode
    main_gpu*: int32
    tensor_split*: ptr cfloat
    progress_callback*: pointer
    progress_callback_user_data*: pointer
    kv_overrides*: pointer
    vocab_only*: bool
    use_mmap*: bool
    use_direct_io*: bool
    use_mlock*: bool
    check_tensors*: bool
    use_extra_bufts*: bool
    no_host*: bool
    no_alloc*: bool

  LlamaContextParams* {.importc: "struct llama_context_params", header: llama_header.} = object
    n_ctx*: uint32
    n_batch*: uint32
    n_ubatch*: uint32
    n_seq_max*: uint32
    n_threads*: int32
    n_threads_batch*: int32
    rope_scaling_type*: LlamaRopeScalingKind
    pooling_type*: LlamaPoolingKind
    attention_type*: LlamaAttentionKind
    flash_attn_type*: LlamaFlashAttnKind
    rope_freq_base*: cfloat
    rope_freq_scale*: cfloat
    yarn_ext_factor*: cfloat
    yarn_attn_factor*: cfloat
    yarn_beta_fast*: cfloat
    yarn_beta_slow*: cfloat
    yarn_orig_ctx*: uint32
    defrag_thold*: cfloat
    cb_eval*: pointer
    cb_eval_user_data*: pointer
    type_k*: cint
    type_v*: cint
    abort_callback*: pointer
    abort_callback_data*: pointer
    embeddings*: bool
    offload_kqv*: bool
    no_perf*: bool
    op_offload*: bool
    swa_full*: bool
    kv_unified*: bool
    samplers*: pointer
    n_samplers*: csize_t

  LlamaPerfContextData* {.importc: "struct llama_perf_context_data", header: llama_header.} = object
    t_start_ms*: cdouble
    t_load_ms*: cdouble
    t_p_eval_ms*: cdouble
    t_eval_ms*: cdouble
    n_p_eval*: int32
    n_eval*: int32
    n_reused*: int32

  LlamaPerfSamplerData* {.importc: "struct llama_perf_sampler_data", header: llama_header.} = object
    t_sample_ms*: cdouble
    n_sample*: int32

# =====================================================================================================================
# Constants
# =====================================================================================================================

const
  LLAMA_DEFAULT_SEED* = 0xFFFFFFFF'u32
  LLAMA_TOKEN_NULL* = -1'i32

# =====================================================================================================================
# Backend
# =====================================================================================================================

proc llama_backend_init*() {.llama_ffi.}
proc llama_backend_free*() {.llama_ffi.}

# =====================================================================================================================
# Default params
# =====================================================================================================================

proc llama_model_default_params*(): LlamaModelParams {.llama_ffi.}
proc llama_context_default_params*(): LlamaContextParams {.llama_ffi.}
proc llama_sampler_chain_default_params*(): LlamaSamplerChainParams {.llama_ffi.}

# =====================================================================================================================
# Model
# =====================================================================================================================

proc llama_model_load_from_file*(path_model: cstring, params: LlamaModelParams): ptr LlamaModel {.llama_ffi.}
proc llama_model_free*(model: ptr LlamaModel) {.llama_ffi.}

proc llama_model_get_vocab*(model: ptr LlamaModel): ptr LlamaVocab {.llama_ffi.}
proc llama_model_n_ctx_train*(model: ptr LlamaModel): int32 {.llama_ffi.}
proc llama_model_n_embd*(model: ptr LlamaModel): int32 {.llama_ffi.}
proc llama_model_n_layer*(model: ptr LlamaModel): int32 {.llama_ffi.}
proc llama_model_n_head*(model: ptr LlamaModel): int32 {.llama_ffi.}
proc llama_model_size*(model: ptr LlamaModel): uint64 {.llama_ffi.}
proc llama_model_n_params*(model: ptr LlamaModel): uint64 {.llama_ffi.}
proc llama_model_desc*(model: ptr LlamaModel, buf: cstring, buf_size: csize_t): int32 {.llama_ffi.}
proc llama_model_chat_template*(model: ptr LlamaModel, name: cstring): cstring {.llama_ffi.}
proc llama_model_has_encoder*(model: ptr LlamaModel): bool {.llama_ffi.}
proc llama_model_has_decoder*(model: ptr LlamaModel): bool {.llama_ffi.}
proc llama_model_is_recurrent*(model: ptr LlamaModel): bool {.llama_ffi.}

proc llama_model_meta_val_str*(model: ptr LlamaModel, key: cstring, buf: cstring, buf_size: csize_t): int32 {.llama_ffi.}
proc llama_model_meta_count*(model: ptr LlamaModel): int32 {.llama_ffi.}

# =====================================================================================================================
# Context
# =====================================================================================================================

proc llama_init_from_model*(model: ptr LlamaModel, params: LlamaContextParams): ptr LlamaContext {.llama_ffi.}
proc llama_free*(ctx: ptr LlamaContext) {.llama_ffi.}

proc llama_n_ctx*(ctx: ptr LlamaContext): uint32 {.llama_ffi.}
proc llama_n_batch*(ctx: ptr LlamaContext): uint32 {.llama_ffi.}
proc llama_n_ubatch*(ctx: ptr LlamaContext): uint32 {.llama_ffi.}
proc llama_n_seq_max*(ctx: ptr LlamaContext): uint32 {.llama_ffi.}

proc llama_get_model*(ctx: ptr LlamaContext): ptr LlamaModel {.llama_ffi.}
proc llama_get_memory*(ctx: ptr LlamaContext): LlamaMemoryT {.llama_ffi.}
when defined(static_llama):
  proc llama_pooling_type_get*(ctx: ptr LlamaContext): LlamaPoolingKind {.cdecl, importc: "llama_pooling_type".}
else:
  proc llama_pooling_type_get*(ctx: ptr LlamaContext): LlamaPoolingKind {.cdecl, dynlib: libllama, importc: "llama_pooling_type".}

# =====================================================================================================================
# Memory (KV cache)
# =====================================================================================================================

proc llama_memory_clear*(mem: LlamaMemoryT, data: bool) {.llama_ffi.}
proc llama_memory_seq_rm*(mem: LlamaMemoryT, seq_id: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): bool {.llama_ffi.}
proc llama_memory_seq_cp*(mem: LlamaMemoryT, seq_id_src: LlamaSeqId, seq_id_dst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) {.llama_ffi.}
proc llama_memory_seq_keep*(mem: LlamaMemoryT, seq_id: LlamaSeqId) {.llama_ffi.}
proc llama_memory_seq_add*(mem: LlamaMemoryT, seq_id: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, delta: LlamaPos) {.llama_ffi.}
proc llama_memory_seq_pos_min*(mem: LlamaMemoryT, seq_id: LlamaSeqId): LlamaPos {.llama_ffi.}
proc llama_memory_seq_pos_max*(mem: LlamaMemoryT, seq_id: LlamaSeqId): LlamaPos {.llama_ffi.}

# =====================================================================================================================
# Decoding
# =====================================================================================================================

proc llama_batch_get_one*(tokens: ptr LlamaToken, n_tokens: int32): LlamaBatch {.llama_ffi.}
proc llama_batch_init*(n_tokens: int32, embd: int32, n_seq_max: int32): LlamaBatch {.llama_ffi.}
proc llama_batch_free*(batch: LlamaBatch) {.llama_ffi.}

proc llama_encode*(ctx: ptr LlamaContext, batch: LlamaBatch): int32 {.llama_ffi.}
proc llama_decode*(ctx: ptr LlamaContext, batch: LlamaBatch): int32 {.llama_ffi.}

proc llama_set_n_threads*(ctx: ptr LlamaContext, n_threads: int32, n_threads_batch: int32) {.llama_ffi.}
proc llama_set_embeddings*(ctx: ptr LlamaContext, embeddings: bool) {.llama_ffi.}
proc llama_set_causal_attn*(ctx: ptr LlamaContext, causal_attn: bool) {.llama_ffi.}
proc llama_synchronize*(ctx: ptr LlamaContext) {.llama_ffi.}

# =====================================================================================================================
# Logits / embeddings
# =====================================================================================================================

proc llama_get_logits*(ctx: ptr LlamaContext): ptr cfloat {.llama_ffi.}
proc llama_get_logits_ith*(ctx: ptr LlamaContext, i: int32): ptr cfloat {.llama_ffi.}
proc llama_get_embeddings*(ctx: ptr LlamaContext): ptr cfloat {.llama_ffi.}
proc llama_get_embeddings_ith*(ctx: ptr LlamaContext, i: int32): ptr cfloat {.llama_ffi.}
proc llama_get_embeddings_seq*(ctx: ptr LlamaContext, seq_id: LlamaSeqId): ptr cfloat {.llama_ffi.}

# =====================================================================================================================
# Vocab
# =====================================================================================================================

when defined(static_llama):
  proc llama_vocab_type_get*(vocab: ptr LlamaVocab): LlamaVocabKind {.cdecl, importc: "llama_vocab_type".}
else:
  proc llama_vocab_type_get*(vocab: ptr LlamaVocab): LlamaVocabKind {.cdecl, dynlib: libllama, importc: "llama_vocab_type".}
proc llama_vocab_n_tokens*(vocab: ptr LlamaVocab): int32 {.llama_ffi.}
proc llama_vocab_get_text*(vocab: ptr LlamaVocab, token: LlamaToken): cstring {.llama_ffi.}
proc llama_vocab_get_score*(vocab: ptr LlamaVocab, token: LlamaToken): cfloat {.llama_ffi.}
proc llama_vocab_get_attr*(vocab: ptr LlamaVocab, token: LlamaToken): LlamaTokenAttr {.llama_ffi.}
proc llama_vocab_is_eog*(vocab: ptr LlamaVocab, token: LlamaToken): bool {.llama_ffi.}
proc llama_vocab_is_control*(vocab: ptr LlamaVocab, token: LlamaToken): bool {.llama_ffi.}
proc llama_vocab_bos*(vocab: ptr LlamaVocab): LlamaToken {.llama_ffi.}
proc llama_vocab_eos*(vocab: ptr LlamaVocab): LlamaToken {.llama_ffi.}
proc llama_vocab_eot*(vocab: ptr LlamaVocab): LlamaToken {.llama_ffi.}
proc llama_vocab_nl*(vocab: ptr LlamaVocab): LlamaToken {.llama_ffi.}
proc llama_vocab_pad*(vocab: ptr LlamaVocab): LlamaToken {.llama_ffi.}
proc llama_vocab_get_add_bos*(vocab: ptr LlamaVocab): bool {.llama_ffi.}
proc llama_vocab_get_add_eos*(vocab: ptr LlamaVocab): bool {.llama_ffi.}

# =====================================================================================================================
# Tokenization
# =====================================================================================================================

proc llama_tokenize*(vocab: ptr LlamaVocab, text: cstring, text_len: int32, tokens: ptr LlamaToken, n_tokens_max: int32, add_special: bool, parse_special: bool): int32 {.llama_ffi.}
proc llama_token_to_piece*(vocab: ptr LlamaVocab, token: LlamaToken, buf: cstring, length: int32, lstrip: int32, special: bool): int32 {.llama_ffi.}
proc llama_detokenize*(vocab: ptr LlamaVocab, tokens: ptr LlamaToken, n_tokens: int32, text: cstring, text_len_max: int32, remove_special: bool, unparse_special: bool): int32 {.llama_ffi.}

# =====================================================================================================================
# Chat templates
# =====================================================================================================================

proc llama_chat_apply_template*(tmpl: cstring, chat: ptr LlamaChatMessage, n_msg: csize_t, add_ass: bool, buf: cstring, length: int32): int32 {.llama_ffi.}

# =====================================================================================================================
# Sampling
# =====================================================================================================================

proc llama_sampler_chain_init*(params: LlamaSamplerChainParams): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_chain_add*(chain: ptr LlamaSampler, smpl: ptr LlamaSampler) {.llama_ffi.}
proc llama_sampler_chain_get*(chain: ptr LlamaSampler, i: int32): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_chain_n*(chain: ptr LlamaSampler): cint {.llama_ffi.}
proc llama_sampler_chain_remove*(chain: ptr LlamaSampler, i: int32): ptr LlamaSampler {.llama_ffi.}

proc llama_sampler_name*(smpl: ptr LlamaSampler): cstring {.llama_ffi.}
proc llama_sampler_accept*(smpl: ptr LlamaSampler, token: LlamaToken) {.llama_ffi.}
proc llama_sampler_apply*(smpl: ptr LlamaSampler, cur_p: ptr LlamaTokenDataArray) {.llama_ffi.}
proc llama_sampler_reset*(smpl: ptr LlamaSampler) {.llama_ffi.}
proc llama_sampler_clone*(smpl: ptr LlamaSampler): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_free*(smpl: ptr LlamaSampler) {.llama_ffi.}
proc llama_sampler_sample*(smpl: ptr LlamaSampler, ctx: ptr LlamaContext, idx: int32): LlamaToken {.llama_ffi.}
proc llama_sampler_get_seed*(smpl: ptr LlamaSampler): uint32 {.llama_ffi.}

# Sampler constructors
proc llama_sampler_init_greedy*(): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_dist*(seed: uint32): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_top_k*(k: int32): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_top_p*(p: cfloat, min_keep: csize_t): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_min_p*(p: cfloat, min_keep: csize_t): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_typical*(p: cfloat, min_keep: csize_t): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_temp*(t: cfloat): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_temp_ext*(t: cfloat, delta: cfloat, exponent: cfloat): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_xtc*(p: cfloat, t: cfloat, min_keep: csize_t, seed: uint32): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_top_n_sigma*(n: cfloat): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_mirostat*(n_vocab: int32, seed: uint32, tau: cfloat, eta: cfloat, m: int32): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_mirostat_v2*(seed: uint32, tau: cfloat, eta: cfloat): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_penalties*(penalty_last_n: int32, penalty_repeat: cfloat, penalty_freq: cfloat, penalty_present: cfloat): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_grammar*(vocab: ptr LlamaVocab, grammar_str: cstring, grammar_root: cstring): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_infill*(vocab: ptr LlamaVocab): ptr LlamaSampler {.llama_ffi.}
proc llama_sampler_init_logit_bias*(n_vocab: int32, n_logit_bias: int32, logit_bias: ptr LlamaLogitBias): ptr LlamaSampler {.llama_ffi.}

# =====================================================================================================================
# Performance
# =====================================================================================================================

proc llama_perf_context*(ctx: ptr LlamaContext): LlamaPerfContextData {.llama_ffi.}
proc llama_perf_context_reset*(ctx: ptr LlamaContext) {.llama_ffi.}
proc llama_perf_sampler*(chain: ptr LlamaSampler): LlamaPerfSamplerData {.llama_ffi.}
proc llama_perf_sampler_reset*(chain: ptr LlamaSampler) {.llama_ffi.}

# =====================================================================================================================
# System info
# =====================================================================================================================

proc llama_print_system_info*(): cstring {.llama_ffi.}
proc llama_max_devices*(): csize_t {.llama_ffi.}
proc llama_supports_mmap*(): bool {.llama_ffi.}
proc llama_supports_mlock*(): bool {.llama_ffi.}
proc llama_supports_gpu_offload*(): bool {.llama_ffi.}
proc llama_time_us*(): int64 {.llama_ffi.}
