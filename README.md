# Embeddings (Go + ONNX Runtime)

A Go module for generating **sentence embeddings** from pre-trained transformer models exported to ONNX.  
This project is motivated by the need to integrate sentence-level semantic search and vector storage into **Go-based pipelines** without depending on Python or external services.

---

## 📚 Background

- **Sentence embeddings**: Dense vectors representing the semantic meaning of text. Useful for search, clustering, retrieval, RAG, etc.
- **Pooling**: Models like BERT output `[batch, seq, hidden]`. To get one vector per sentence, token embeddings should be mean-pooled and normalized.
- **ONNX Runtime**: A cross-platform inference engine that runs exported transformer models efficiently on CPU/GPU.
- **Tokenizers**: Hugging Face JSON or SentencePiece models used to convert text to token IDs.

---

## 🏛️ Architecture

- **Tokenizer**: Uses [`sugarme/tokenizer`](https://github.com/sugarme/tokenizer) to load Hugging Face tokenizers.
- **ONNX Runtime**: Loaded via [`yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go).
- **Encoder**: Handles inputs/outputs, pooling, and normalization.
- **Vector DB integration**: Embeddings are ready to be indexed into Elasticsearch or similar stores.

---

## ⚙️ Setup

### Prerequisites
- Go 1.22+
- ONNX Runtime 1.22.x shared library installed (CPU build is fine)
- Define `ORT_VER` & `ORT_ARCH` in makefile & run `make setup-ort` to setup onnx runtime for defined version & architecture
- Set/export `ONNXRUNTIME_SHARED_LIBRARY_PATH` for selected ort version
- Get the module `go get github.com/hankgalt/embeddings`
- Download & setup ONNX compliant model & tokenizer. See: https://huggingface.co/onnx-models

### Dev Notes
- `Launch Embeddings Example` from VSCode debug targets

## 🙌 Acknowledgements
- ONNX Runtime
- yalue/onnxruntime_go
- sugarme/tokenizer
- Hugging Face & SentenceTransformers communi