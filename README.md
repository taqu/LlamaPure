# LlamaPure

**A dependency-free .NET Standard 2.0 P/Invoke wrapper for llama.cpp (CPU inference, Completion, and Embeddings).**

[![NuGet](https://img.shields.io/nuget/v/LlamaPure.svg)](https://www.nuget.org/packages/LlamaPure)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![.NET Standard](https://img.shields.io/badge/.NET%20Standard-2.0-blue.svg)](https://docs.microsoft.com/en-us/dotnet/standard/net-standard)

---

## Design Philosophy

LlamaPure is designed for environments where the addition of an extensive package dependency graph is restricted or undesirable — such as large legacy enterprise systems, strict compliance-driven build pipelines, or projects where tight control over every binary in the deployment package is required.

The library achieves this through a single architectural choice: a direct, thin P/Invoke layer over the official `llama.dll`. It introduces no third-party NuGet dependencies and references nothing beyond `System` and `System.Runtime.InteropServices`. The public API is intentionally narrow, covering only the two inference operations most commonly needed in line-of-business applications:

- **Text Completion** — autoregressive token generation from a prompt
- **Text Embedding** — dense float-vector extraction from text

Everything else is left to the caller, keeping the integration footprint as small as possible.

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| Runtime   | .NET Framework 4.6.1 / 4.7.2 / 4.8, or .NET 6 / 7 / 8 / 9 / 10 |
| Platform  | Windows x64 |
| Backend   | CPU (llama.cpp CPU build) |

---

## Installation

### Add the NuGet package

```
dotnet add package LlamaPure
```

Or via the Package Manager Console:

```
Install-Package LlamaPure
```

---

## Quick Start

### Text Completion

```csharp
using LlamaPure;

using (var client = new LlamaPureClient(@"C:\models\llama-3.2-1b.Q4_K_M.gguf"))
{
    string result = client.Complete("The capital of France is", maxNewTokens: 32);
    Console.WriteLine(result);
}
```

### Text Embedding

```csharp
using LlamaPure;

using (var client = new LlamaPureClient(@"C:\models\nomic-embed-text.Q4_K_M.gguf"))
{
    float[] vector = client.GetEmbedding("Hello, world!");
    Console.WriteLine("Embedding dimensions: " + vector.Length);
}
```

### Configuring Context and Threads

```csharp
using LlamaPure;

using (var client = new LlamaPureClient(
    modelPath: @"C:\models\llama-3.2-1b.Q4_K_M.gguf",
    contextSize: 4096,
    threads: 8))
{
    string answer = client.Complete("Explain the Pythagorean theorem.", maxNewTokens: 128);
    Console.WriteLine(answer);
}
```

All examples use `string` and standard managed arrays and are compatible with C# 6.0 and later.

---

## API Reference

### `LlamaPureClient`

```csharp
public sealed class LlamaPureClient : IDisposable
```

#### Constructor

```csharp
LlamaPureClient(string modelPath, uint contextSize = 2048, int threads = 4)
```

Initializes the llama backend, loads the GGUF model, and creates an inference context. Throws `InvalidOperationException` if the model cannot be loaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelPath` | `string` | — | Absolute or relative path to a `.gguf` model file |
| `contextSize` | `uint` | `2048` | Maximum context length in tokens (KV-cache size) |
| `threads` | `int` | `4` | Number of CPU threads used during generation |

#### `Complete`

```csharp
string Complete(string prompt, int maxNewTokens = 128)
```

Tokenizes `prompt`, performs a prefill decode, then generates up to `maxNewTokens` tokens using greedy sampling. Generation stops early when an end-of-generation token is produced. Returns only the generated continuation; the prompt text is not included in the return value.

#### `GetEmbedding`

```csharp
float[] GetEmbedding(string text)
```

Tokenizes `text`, executes a single forward pass with embeddings enabled, and returns the last-token hidden state as a `float[]` of length equal to the model's embedding dimension (`n_embd`). Intended for use with embedding-capable models.

#### `Dispose`

```csharp
void Dispose()
```

Releases the inference context and model, and frees the llama backend. Idempotent — safe to call multiple times.

---

## Building from Source

```bat
git clone --recurse-submodules https://github.com/taqu/LlamaPure
cd LlamaPure
dotnet build LlamaPure/LlamaPure.csproj -c Release
```

To build the native DLLs from the included llama.cpp submodule:

```bat
cd LlamaWrapper
build.bat
```

---

## License
This software is distributed under two licenses 'The MIT License' or 'Public Domain', choose whichever you like.
