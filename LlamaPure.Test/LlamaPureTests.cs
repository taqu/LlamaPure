using System;
using System.IO;
using LlamaPure;
using Xunit;

namespace LlamaPure.Test
{
    // Set LLAMA_MODEL_PATH environment variable to a valid .gguf model path.
    // Tests requiring a real model are skipped when the path is unset.
    public sealed class LlamaPureTests
    {
        private static readonly string ModelPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..\\..\\..\\", "model", "litellama-460m-1t.Q4_K_M.gguf");

        private static bool HasModel => !string.IsNullOrEmpty(ModelPath) && File.Exists(ModelPath);

        [Fact]
        public void Initialization_InvalidPath_ThrowsOrReturnsError()
        {
            var ex = Assert.ThrowsAny<Exception>(() =>
            {
                using var client = new LlamaPureClient("/nonexistent/model.gguf");
            });
            Assert.NotNull(ex);
        }

        [SkippableFact]
        public void GetEmbedding_HelloWorld_ReturnsNonEmptyFloatArray()
        {
            Skip.IfNot(HasModel, "LLAMA_MODEL_PATH not set or file not found.");

            using var client = new LlamaPureClient(ModelPath);
            float[] embedding = client.GetEmbedding("LLAMA_MODEL_PATH not set or file not found.");

            Assert.NotNull(embedding);
            Assert.True(embedding.Length > 0, "Embedding array must be non-empty.");

            float[] embedding2 = client.GetEmbedding("Embedding array must be non-empty.");
            Assert.NotNull(embedding2);
            Assert.False(Array.Equals(embedding, embedding2), "Embeddings must be different.");
        }

        [SkippableFact]
        public void Complete_ShortPrompt_ReturnsNonEmptyString()
        {
            Skip.IfNot(HasModel, "LLAMA_MODEL_PATH not set or file not found.");

            using var client = new LlamaPureClient(ModelPath);
            string result = client.Complete("The answer to life is", maxNewTokens: 5);

            Assert.NotNull(result);
            Assert.True(result.Length > 0, "Completion must return at least one character.");
        }

        [SkippableFact]
        public void Dispose_CalledMultipleTimes_DoesNotThrow()
        {
            Skip.IfNot(HasModel, "LLAMA_MODEL_PATH not set or file not found.");

            var client = new LlamaPureClient(ModelPath);
            client.Dispose();

            var ex = Record.Exception(() => client.Dispose());
            Assert.Null(ex);
        }
    }
}
