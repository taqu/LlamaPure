# Task: Fix Embedding Extraction for Embedding Models in `LlamaPureClient`

## Background

`LlamaPureClient.GetEmbedding(string text)` currently works incorrectly with embedding-only GGUF models.

The current implementation:

* Clears the KV cache.
* Tokenizes the input.
* Calls `llama_decode()`.
* Retrieves the embedding using:

```cpp
llama_get_embeddings_ith(ctx, last_token_index)
```

For embedding models this frequently returns an all-zero vector.

The implementation currently assumes that embeddings are exposed in the same way as autoregressive language models, which is not always true in recent versions of llama.cpp.

---

## Objective

Modify the embedding implementation so that it works correctly with both:

* standard decoder LLMs that expose token embeddings
* embedding-only GGUF models

The implementation should follow the current llama.cpp API instead of relying on assumptions about internal behavior.

---

## Requirements

### 1. Investigate the current llama.cpp embedding API

Read the current llama.cpp implementation and determine:

* how embedding mode is enabled
* whether embeddings are sequence-level or token-level
* which API is intended for embedding models
* whether `llama_get_embeddings()` should be used instead of `llama_get_embeddings_ith()`
* whether pooling must be enabled during context creation
* whether the batch must mark a sequence as completed
* whether `embeddings=true` or another context parameter is required

Do **not** assume that the current implementation is correct.

---

### 2. Verify context creation

Inspect the code that creates the llama context.

Confirm whether embedding models require configuration such as:

* embeddings enabled
* pooling type
* sequence embedding mode
* other embedding-specific context parameters

If additional configuration is required, implement it.

---

### 3. Verify batch construction

Inspect `FillBatch()` carefully.

Verify that:

* token positions are correct
* sequence IDs are correct
* logits flags are correct
* embedding requests are correctly represented
* the final token (or sequence) is properly marked according to llama.cpp requirements

If the batch construction is incorrect for embedding models, fix it.

---

### 4. Retrieve embeddings correctly

Replace the retrieval logic if necessary.

Possible APIs include (depending on llama.cpp version):

* `llama_get_embeddings()`
* `llama_get_embeddings_seq()`
* `llama_get_embeddings_ith()`

Choose the API recommended by the current llama.cpp implementation.

Do not keep using `llama_get_embeddings_ith()` if it is not appropriate for embedding models.

---

### 5. Support both model types

The implementation should continue to work for:

* decoder language models
* embedding-only models

If different retrieval paths are required, detect the model capabilities and use the correct path.

Avoid duplicate code where possible.

---

### 6. Improve diagnostics

If embedding retrieval fails:

* report whether the model supports embeddings
* report whether the returned pointer is null
* report whether pooling is enabled
* report embedding dimension
* provide meaningful exception messages

Do not silently return a zero vector.

---

### 7. Validation

Test with at least:

* an embedding GGUF model
* a normal instruct/chat GGUF model

Confirm that:

* returned vectors are non-zero
* vector length equals `_nEmbd`
* repeated calls return deterministic embeddings
* cosine similarity behaves as expected for similar sentences

---

## Constraints

* Keep the public API unchanged.
* Preserve existing behavior for language models whenever possible.
* Follow the latest llama.cpp API and coding conventions.
* Do not introduce unnecessary abstractions.
* Keep the implementation simple and maintainable.

## Deliverables

1. Updated `LlamaPureClient.cs`
2. Updated `LlamaPureNative.cs` if additional native bindings are required
3. A brief summary explaining:

   * the root cause,
   * the llama.cpp API changes involved,
   * why the previous implementation produced zero vectors,
   * how the new implementation works.
