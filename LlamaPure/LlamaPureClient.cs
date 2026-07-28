using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaPure
{
    public sealed class LlamaPureClient : IDisposable
    {
        private IntPtr _model;
        private IntPtr _ctx;
        private IntPtr _vocab;
        private int _nVocab;
        private int _nEmbd;
        private bool _disposed;
        private readonly int _maxChunkSize;

        public LlamaPureClient(string modelPath, uint contextSize = 2048, int threads = 4)
        {
            if (string.IsNullOrEmpty(modelPath)) {
                throw new ArgumentException("Model path must not be null or empty.", nameof(modelPath));
            }
            _maxChunkSize = (int)contextSize;

            LlamaPureNative.llama_backend_init();

            LlamaPureNative.LlamaModelParams modelParams = LlamaPureNative.llama_model_default_params();
            modelParams.load_mode = LlamaPureNative.LlamaLoadMode.None;
            _model = LlamaPureNative.llama_model_load_from_file(modelPath, modelParams);
            if (_model == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to load model from '{modelPath}'.");

            _vocab = LlamaPureNative.llama_model_get_vocab(_model);
            if (_vocab == IntPtr.Zero)
                throw new InvalidOperationException("Failed to get vocabulary from model.");

            _nVocab = LlamaPureNative.llama_vocab_n_tokens(_vocab);
            _nEmbd  = LlamaPureNative.llama_model_n_embd(_model);

            var ctxParams = LlamaPureNative.llama_context_default_params();
            ctxParams.n_ctx       = contextSize;
            ctxParams.n_batch     = contextSize;
            ctxParams.n_threads   = threads;
            ctxParams.embeddings  = 1;

            _ctx = LlamaPureNative.llama_init_from_model(_model, ctxParams);
            if (_ctx == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create llama context.");
        }

        public List<float[]> GetEmbedding(string text, int overlap = 128)
        {
            ThrowIfDisposed();
            if (text == null){
                throw new ArgumentNullException(nameof(text));
            }
            IntPtr mem = LlamaPureNative.llama_get_memory(_ctx);
            LlamaPureNative.llama_memory_clear(mem, 0);
            int chunkSize = _maxChunkSize;
            if (overlap >= chunkSize) {
                throw new ArgumentException("Overlap size must be smaller than the allowed context chunk size.");
            }

            int[] tokenIds = Tokenize(text, addSpecial: true);
            List<float[]> embeddingsList = new List<float[]>();
            if (tokenIds.Length == 0) {
                return embeddingsList;
            }

            //int[] chunkTokens = new int[_maxChunkSize];
            int index = 0;
            while (index < tokenIds.Length)
            {
                int currentChunkSize = Math.Min(chunkSize, tokenIds.Length - index);
                int[] chunkTokens = new int[currentChunkSize];
                Array.Copy(tokenIds, index, chunkTokens, 0, currentChunkSize);

                float[] chunkVector = ProcessSingleTokenChunk(chunkTokens, currentChunkSize);
                embeddingsList.Add(chunkVector);

                index += chunkSize - overlap;

                // Break if we hit the natural end of the text stream
                if (index >= tokenIds.Length || currentChunkSize < chunkSize) {
                    break;
                }
            }

            return embeddingsList;
        }

        private float[] ProcessSingleTokenChunk(int[] tokenIds, int length)
        {
            // Reset state container inside llama context memory block
            IntPtr mem = LlamaPureNative.llama_get_memory(_ctx);
            LlamaPureNative.llama_memory_clear(mem, 0);

            // Sequence ID allocation is managed cleanly inside FillBatch (Mapped to Seq ID: 0)
            LlamaPureNative.LlamaBatch batch = LlamaPureNative.llama_batch_init(length, 0, 1);
            try
            {
                // Logits flag processing optimized: pass false for embedding models to avoid tracking redundant data
                FillBatch(ref batch, tokenIds, length, posOffset: 0, setLogits: false);

                // Check if model has an encoder architecture or if pooling context is activated
                bool hasEncoder = LlamaPureNative.llama_model_has_encoder(_model);
                LlamaPureNative.LlamaPoolingType poolingType = LlamaPureNative.llama_pooling_type(_ctx);
                bool isPureEmbeddingModel = (poolingType != LlamaPureNative.LlamaPoolingType.None);

                int ret;
                if (hasEncoder && isPureEmbeddingModel)
                {
                    ret = LlamaPureNative.llama_encode(_ctx, batch);
                }
                else
                {
                    // Fallback route targeting standard causal autoregressive models
                    ret = LlamaPureNative.llama_decode(_ctx, batch);
                }

                if (ret != 0)
                    throw new InvalidOperationException($"Native execution failed during inference execution with error code: {ret}.");

                IntPtr embdPtr;
                if (poolingType != LlamaPureNative.LlamaPoolingType.None)
                {
                    // Get the pooled sequence embedding vector from Sequence ID: 0
                    embdPtr = LlamaPureNative.llama_get_embeddings_seq(_ctx, 0);
                    if (embdPtr == IntPtr.Zero)
                        throw new InvalidOperationException("llama_get_embeddings_seq returned a null pointer.");
                }
                else
                {
                    // Fallback option utilizing last token calculation if pooling is unsupported
                    embdPtr = LlamaPureNative.llama_get_embeddings_ith(_ctx, tokenIds.Length - 1);
                    if (embdPtr == IntPtr.Zero)
                        throw new InvalidOperationException("llama_get_embeddings_ith returned a null pointer.");
                }

                var result = new float[_nEmbd];
                Marshal.Copy(embdPtr, result, 0, _nEmbd);
                return result;
            }
            finally
            {
                // Guaranteed cleanup prevents out-of-memory errors over long continuous document chunks
                LlamaPureNative.llama_batch_free(batch);
            }
        }

        public string Complete(string prompt, int maxNewTokens = 128)
        {
            ThrowIfDisposed();
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));
            if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));

            IntPtr mem = LlamaPureNative.llama_get_memory(_ctx);
            LlamaPureNative.llama_memory_clear(mem, 0);

            int[] promptTokens = Tokenize(prompt, addSpecial: true);
            if (promptTokens.Length == 0)
                throw new InvalidOperationException("Tokenization produced no tokens.");

            // Prefill: decode all prompt tokens, request logits for the last one only.
            LlamaPureNative.LlamaBatch batch = LlamaPureNative.llama_batch_init(promptTokens.Length, 0, 1);
            int prefillRet;
            try
            {
                FillBatch(ref batch, promptTokens, posOffset: 0, setLastLogit: true);
                prefillRet = LlamaPureNative.llama_decode(_ctx, batch);
            }
            finally
            {
                LlamaPureNative.llama_batch_free(batch);
            }

            if (prefillRet != 0)
                throw new InvalidOperationException($"llama_decode (prefill) failed with code {prefillRet}.");

            // Sample first token from the last prompt token's logits.
            IntPtr logitsPtr = LlamaPureNative.llama_get_logits_ith(_ctx, promptTokens.Length - 1);
            int sampledToken = GreedySample(logitsPtr, _nVocab);

            var sb = new StringBuilder();
            int currentPos = promptTokens.Length;

            for (int step = 0; step < maxNewTokens; step++)
            {
                if (IsEndOfGeneration(sampledToken))
                    break;

                sb.Append(TokenToString(sampledToken));

                // Decode the single new token.
                LlamaPureNative.LlamaBatch stepBatch = LlamaPureNative.llama_batch_init(1, 0, 1);
                int stepRet;
                try
                {
                    FillBatch(ref stepBatch, new[] { sampledToken }, posOffset: currentPos, setLastLogit: true);
                    stepRet = LlamaPureNative.llama_decode(_ctx, stepBatch);
                }
                finally
                {
                    LlamaPureNative.llama_batch_free(stepBatch);
                }

                if (stepRet != 0)
                    throw new InvalidOperationException($"llama_decode (step {step}) failed with code {stepRet}.");

                currentPos++;
                logitsPtr   = LlamaPureNative.llama_get_logits_ith(_ctx, 0);
                sampledToken = GreedySample(logitsPtr, _nVocab);
            }

            return sb.ToString();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_ctx != IntPtr.Zero)
            {
                LlamaPureNative.llama_free(_ctx);
                _ctx = IntPtr.Zero;
            }

            if (_model != IntPtr.Zero)
            {
                LlamaPureNative.llama_model_free(_model);
                _model = IntPtr.Zero;
            }

            LlamaPureNative.llama_backend_free();
        }

        private int[] Tokenize(string text, bool addSpecial)
        {
            int nBytes = Encoding.UTF8.GetByteCount(text);
            int maxTokens = nBytes + 16;
            var tokens = new int[maxTokens];

            byte specialFlag = addSpecial ? (byte)1 : (byte)0;
            int count = LlamaPureNative.llama_tokenize(
                _vocab, text, nBytes, tokens, maxTokens, specialFlag, 0);

            if (count < 0)
            {
                // Buffer too small; reallocate with the exact required size.
                maxTokens = -count;
                tokens = new int[maxTokens];
                count = LlamaPureNative.llama_tokenize(
                    _vocab, text, nBytes, tokens, maxTokens, specialFlag, 0);
            }

            if (count < 0)
                throw new InvalidOperationException($"Tokenization failed (code {count}).");

            var result = new int[count];
            Array.Copy(tokens, result, count);
            return result;
        }

        private void FillBatch(ref LlamaPureNative.LlamaBatch batch, int[] tokenIds, int length, int posOffset, bool setLogits)
        {
            int n = length;
            batch.n_tokens = n;

            for (int i = 0; i < n; i++)
            {
                Marshal.WriteInt32(batch.token + i * 4, tokenIds[i]);
                Marshal.WriteInt32(batch.pos + i * 4, posOffset + i);
                Marshal.WriteInt32(batch.n_seq_id + i * 4, 1);

                IntPtr seqIdRow = Marshal.ReadIntPtr(batch.seq_id + i * IntPtr.Size);
                Marshal.WriteInt32(seqIdRow, 0);

                byte logitFlag = setLogits ? (byte)1 : (byte)0;
                Marshal.WriteByte(batch.logits + i, logitFlag);
            }
        }

        private void FillBatch(ref LlamaPureNative.LlamaBatch batch, int[] tokenIds, int posOffset, bool setLastLogit, bool setAllLogits = false)
        {
            int n = tokenIds.Length;
            batch.n_tokens = n;

            for (int i = 0; i < n; i++)
            {
                Marshal.WriteInt32(batch.token + i * 4, tokenIds[i]);
                Marshal.WriteInt32(batch.pos + i * 4, posOffset + i);
                Marshal.WriteInt32(batch.n_seq_id + i * 4, 1);
                IntPtr seqIdRow = Marshal.ReadIntPtr(batch.seq_id + i * IntPtr.Size);
                Marshal.WriteInt32(seqIdRow, 0);
                byte logitFlag = (setAllLogits || (setLastLogit && i == n - 1)) ? (byte)1 : (byte)0;
                Marshal.WriteByte(batch.logits + i, logitFlag);
            }
        }

        private static int GreedySample(IntPtr logitsPtr, int nVocab)
        {
            if (logitsPtr == IntPtr.Zero)
                throw new InvalidOperationException("Logits pointer is null.");

            int bestToken = 0;
            float bestLogit = ReadFloat(logitsPtr, 0);

            for (int i = 1; i < nVocab; i++)
            {
                float logit = ReadFloat(logitsPtr, i);
                if (logit > bestLogit)
                {
                    bestLogit = logit;
                    bestToken = i;
                }
            }

            return bestToken;
        }

        private string TokenToString(int token)
        {
            var buf = new byte[256];
            int len = LlamaPureNative.llama_token_to_piece(_vocab, token, buf, buf.Length, 0, 0);
            if (len <= 0) return string.Empty;
            return Encoding.UTF8.GetString(buf, 0, len);
        }

        private bool IsEndOfGeneration(int token)
        {
            return LlamaPureNative.llama_vocab_is_eog(_vocab, token) != 0;
        }

        private static float ReadFloat(IntPtr ptr, int index)
        {
            // Read 4 bytes and reinterpret as float without unsafe code.
            int bits = Marshal.ReadInt32(ptr + index * 4);
            return Int32BitsToFloat(bits);
        }

        private static float Int32BitsToFloat(int value)
        {
            // BitConverter.ToSingle on a 4-byte stack allocation avoids unsafe.
            byte[] bytes = BitConverter.GetBytes(value);
            return BitConverter.ToSingle(bytes, 0);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LlamaPureClient));
        }
    }
}
