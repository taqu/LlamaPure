using System;
using System.Runtime.InteropServices;

public static class LlamaPureNative
{
    private const string LlamaDll = "llama";

    public enum LlamaSplitMode : int
    {
        None   = 0,
        Layer  = 1,
        Row    = 2,
        Tensor = 3,
    }

    public enum LlamaLoadMode : int
    {
        Auto      = -1,
        None      =  0,
        Mmap      =  1,
        Mlock     =  2,
        MmapMlock =  3,
        DirectIo  =  4,
    }

    public enum LlamaLazyMode : int
    {
        Off  = 0,
        Auto = 1,
        On   = 2,
    }

    public enum LlamaContextType : int
    {
        Default = 0,
        Mtp     = 1,
    }

    public enum LlamaRopeScalingType : int
    {
        Unspecified = -1,
        None        = 0,
        Linear      = 1,
        Yarn        = 2,
        LongRope    = 3,
    }

    public enum LlamaPoolingType : int
    {
        Unspecified = -1,
        None        = 0,
        Mean        = 1,
        Cls         = 2,
        Last        = 3,
        Rank        = 4,
    }

    public enum LlamaAttentionType : int
    {
        Unspecified = -1,
        Causal      = 0,
        NonCausal   = 1,
    }

    public enum LlamaFlashAttnType : int
    {
        Auto     = -1,
        Disabled = 0,
        Enabled  = 1,
    }

    public enum GgmlType : int
    {
        F32     = 0,
        F16     = 1,
        Q4_0    = 2,
        Q4_1    = 3,
        Q5_0    = 6,
        Q5_1    = 7,
        Q8_0    = 8,
        Q8_1    = 9,
        Q2_K    = 10,
        Q3_K    = 11,
        Q4_K    = 12,
        Q5_K    = 13,
        Q6_K    = 14,
        Q8_K    = 15,
        IQ2_XXS = 16,
        IQ2_XS  = 17,
        IQ3_XXS = 18,
        IQ1_S   = 19,
        IQ4_NL  = 20,
        IQ3_S   = 21,
        IQ2_S   = 22,
        IQ4_XS  = 23,
        I8      = 24,
        I16     = 25,
        I32     = 26,
        I64     = 27,
        F64     = 28,
        IQ1_M   = 29,
        BF16    = 30,
        TQ1_0   = 34,
        TQ2_0   = 35,
        MXFP4   = 39,
        NVFP4   = 40,
        Q1_0    = 41,
        Q2_0    = 42,
        F8_E4M3 = 43,
        Count   = 44,
    }

    // llama_model_params matches the C struct layout on x64:
    //   offset 0:  devices (ptr 8)
    //   offset 8:  tensor_buft_overrides (ptr 8)
    //   offset 16: n_gpu_layers (int32 4)
    //   offset 20: split_mode (enum/int32 4)
    //   offset 24: load_mode (enum/int32 4)
    //   offset 28: lazy_mode (enum/int32 4)
    //   offset 32: main_gpu (int32 4)
    //   offset 36: _pad (4 bytes, aligns tensor_split to 8)
    //   offset 40: tensor_split (ptr 8)
    //   offset 48: progress_callback (ptr 8)
    //   offset 56: progress_callback_user_data (ptr 8)
    //   offset 64: kv_overrides (ptr 8)
    //   offset 72: vocab_only (byte 1)
    //   offset 73: check_tensors (byte 1)
    //   offset 74: use_extra_bufts (byte 1)
    //   offset 75: no_host (byte 1)
    //   offset 76: no_alloc (byte 1)
    //   offset 77: load_mtp (byte 1)
    //   offset 78-79: padding (2 bytes)
    //   total: 80 bytes
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaModelParams
    {
        public IntPtr devices;
        public IntPtr tensor_buft_overrides;
        public int n_gpu_layers;
        public LlamaSplitMode split_mode;
        public LlamaLoadMode load_mode;
        public LlamaLazyMode lazy_mode;
        public int main_gpu;
        public IntPtr tensor_split;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
        public IntPtr kv_overrides;
        public byte vocab_only;
        public byte check_tensors;
        public byte use_extra_bufts;
        public byte no_host;
        public byte no_alloc;
        public byte load_mtp;
    }

    // llama_context_params matches the C struct layout on x64:
    //   offset 0:   n_ctx (uint32 4)
    //   offset 4:   n_batch (uint32 4)
    //   offset 8:   n_ubatch (uint32 4)
    //   offset 12:  n_seq_max (uint32 4)
    //   offset 16:  n_rs_seq (uint32 4)
    //   offset 20:  n_outputs_max (uint32 4)
    //   offset 24:  n_outputs_max_per_seq (uint32 4)
    //   offset 28:  n_threads (int32 4)
    //   offset 32:  n_threads_batch (int32 4)
    //   offset 36:  ctx_type (enum/int32 4)
    //   offset 40:  rope_scaling_type (enum/int32 4)
    //   offset 44:  pooling_type (enum/int32 4)
    //   offset 48:  attention_type (enum/int32 4)
    //   offset 52:  flash_attn_type (enum/int32 4)
    //   offset 56:  rope_freq_base (float 4)
    //   offset 60:  rope_freq_scale (float 4)
    //   offset 64:  yarn_ext_factor (float 4)
    //   offset 68:  yarn_attn_factor (float 4)
    //   offset 72:  yarn_beta_fast (float 4)
    //   offset 76:  yarn_beta_slow (float 4)
    //   offset 80:  yarn_orig_ctx (uint32 4)
    //   offset 84:  _pad (4 bytes, aligns cb_eval to 8)
    //   offset 88:  cb_eval (ptr 8)
    //   offset 96:  cb_eval_user_data (ptr 8)
    //   offset 104: type_k (enum/int32 4)
    //   offset 108: type_v (enum/int32 4)
    //   offset 112: moe_cache_size (size_t 8)
    //   offset 120: moe_cache_cpu_share (float 4)
    //   offset 124: _pad (4 bytes, aligns abort_callback to 8)
    //   offset 128: abort_callback (ptr 8)
    //   offset 136: abort_callback_data (ptr 8)
    //   offset 144: embeddings (byte 1)
    //   offset 145: offload_kqv (byte 1)
    //   offset 146: no_perf (byte 1)
    //   offset 147: op_offload (byte 1)
    //   offset 148: swa_full (byte 1)
    //   offset 149: kv_unified (byte 1)
    //   offset 150: _pad2 (2 bytes, aligns samplers to 8)
    //   offset 152: samplers (ptr 8)
    //   offset 160: n_samplers (size_t 8)
    //   offset 168: ctx_other (ptr 8)
    //   total: 176 bytes
    [StructLayout(LayoutKind.Explicit, Size = 176)]
    public struct LlamaContextParams
    {
        [FieldOffset(0)]   public uint n_ctx;
        [FieldOffset(4)]   public uint n_batch;
        [FieldOffset(8)]   public uint n_ubatch;
        [FieldOffset(12)]  public uint n_seq_max;
        [FieldOffset(16)]  public uint n_rs_seq;
        [FieldOffset(20)]  public uint n_outputs_max;
        [FieldOffset(24)]  public uint n_outputs_max_per_seq;
        [FieldOffset(28)]  public int n_threads;
        [FieldOffset(32)]  public int n_threads_batch;
        [FieldOffset(36)]  public LlamaContextType ctx_type;
        [FieldOffset(40)]  public LlamaRopeScalingType rope_scaling_type;
        [FieldOffset(44)]  public LlamaPoolingType pooling_type;
        [FieldOffset(48)]  public LlamaAttentionType attention_type;
        [FieldOffset(52)]  public LlamaFlashAttnType flash_attn_type;
        [FieldOffset(56)]  public float rope_freq_base;
        [FieldOffset(60)]  public float rope_freq_scale;
        [FieldOffset(64)]  public float yarn_ext_factor;
        [FieldOffset(68)]  public float yarn_attn_factor;
        [FieldOffset(72)]  public float yarn_beta_fast;
        [FieldOffset(76)]  public float yarn_beta_slow;
        [FieldOffset(80)]  public uint yarn_orig_ctx;
        [FieldOffset(88)]  public IntPtr cb_eval;
        [FieldOffset(96)]  public IntPtr cb_eval_user_data;
        [FieldOffset(104)] public GgmlType type_k;
        [FieldOffset(108)] public GgmlType type_v;
        [FieldOffset(112)] public UIntPtr moe_cache_size;
        [FieldOffset(120)] public float moe_cache_cpu_share;
        [FieldOffset(128)] public IntPtr abort_callback;
        [FieldOffset(136)] public IntPtr abort_callback_data;
        [FieldOffset(144)] public byte embeddings;
        [FieldOffset(145)] public byte offload_kqv;
        [FieldOffset(146)] public byte no_perf;
        [FieldOffset(147)] public byte op_offload;
        [FieldOffset(148)] public byte swa_full;
        [FieldOffset(149)] public byte kv_unified;
        [FieldOffset(152)] public IntPtr samplers;
        [FieldOffset(160)] public UIntPtr n_samplers;
        [FieldOffset(168)] public IntPtr ctx_other;
    }

    // llama_batch: n_tokens + 6 pointers = 4 + 6*8 = 52 bytes + 4 pad = 56 bytes
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaBatch
    {
        public int n_tokens;
        public IntPtr token;    // llama_token*   (int32_t*)
        public IntPtr embd;     // float*
        public IntPtr pos;      // llama_pos*     (int32_t*)
        public IntPtr n_seq_id; // int32_t*
        public IntPtr seq_id;   // llama_seq_id** (int32_t**)
        public IntPtr logits;   // int8_t*
    }

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_backend_init();

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_backend_free();

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern LlamaModelParams llama_model_default_params();

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern LlamaContextParams llama_context_default_params();

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false)]
    public static extern IntPtr llama_model_load_from_file(string path_model, LlamaModelParams @params);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_model_free(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_init_from_model(IntPtr model, LlamaContextParams @params);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_free(IntPtr ctx);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool llama_model_has_encoder(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool llama_model_has_decoder(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_model_get_vocab(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_vocab_n_tokens(IntPtr vocab);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_model_n_ctx_train(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_model_n_embd(IntPtr model);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_vocab_bos(IntPtr vocab);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_vocab_eos(IntPtr vocab);

    // bool params are C99 _Bool (1 byte); marshal as byte to avoid BOOL (4-byte) default
    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern byte llama_vocab_is_eog(IntPtr vocab, int token);

    // bool params are C99 _Bool (1 byte); marshal as byte to avoid BOOL (4-byte) default
    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false)]
    public static unsafe  extern int llama_tokenize(
        IntPtr vocab,
        byte* text,
        int text_len,
        int[] tokens,
        int n_tokens_max,
        byte add_special,
        byte parse_special);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_token_to_piece(
        IntPtr vocab,
        int token,
        byte[] buf,
        int length,
        int lstrip,
        byte special);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern LlamaBatch llama_batch_init(int n_tokens, int embd, int n_seq_max);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_batch_free(LlamaBatch batch);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_encode(IntPtr ctx, LlamaBatch batch);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int llama_decode(IntPtr ctx, LlamaBatch batch);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_get_logits(IntPtr ctx);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_get_logits_ith(IntPtr ctx, int i);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_get_embeddings_ith(IntPtr ctx, int i);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr llama_get_embeddings_seq(IntPtr ctx, int seq_id);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern LlamaPoolingType llama_pooling_type(IntPtr ctx);

    // -------------------------------------------------------------------------
    // Memory / KV cache
    // -------------------------------------------------------------------------
    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr llama_get_memory(IntPtr ctx);

    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void llama_memory_clear(IntPtr mem, byte data);

    // Removes tokens in position range [p0, p1) for the given sequence.
    // Pass p1 = -1 to remove from p0 to the end.
    // Returns false if the operation is not supported by this memory type.
    [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool llama_memory_seq_rm(IntPtr mem, int seq_id, int p0, int p1);

}
