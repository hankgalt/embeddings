# Embeddings

### Dev Notes
- Define `ORT_VER` & `ORT_ARCH` in makefile & run `make setup-ort` to setup onnx runtime for mac-arm64
- Set `ONNXRUNTIME_SHARED_LIBRARY_PATH=../runtime/onnxruntime/lib/libonnxruntime.so.{ORT_VER}` with selected ort version in `env/test.env` 
- Download & setup ONNX compliant model & tokenizer. See: https://huggingface.co/onnx-models
- `Launch Embeddings Example` from VSCode debug targets