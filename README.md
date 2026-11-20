# Sentence embeddings from scratch with ONNX Runtime

A step-by-step notebook that rebuilds `SentenceTransformer.encode()` by hand, **without PyTorch**, using only
`onnxruntime`, `tokenizers` and `numpy`. Its output matches sentence-transformers to within ~1e-7.

```python
# instead of this (pulls in PyTorch, ~580 MB)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# you end up with this (~50 MB of dependencies), same vectors
model = OnnxEmbedder("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(["Can I still join the course?", "How do I install Docker?"])
```

The goal is understanding, not a library: every step is printed, explained and checked.

## What you'll learn

| Part | Topic |
|---|---|
| NumPy crash course | Only what's needed: shapes, axes, broadcasting, `@`, `argsort`. No prior NumPy needed |
| Step 0 | Downloading a model and reading the "recipe" files sentence-transformers follows |
| Step 1 | Tokenization: WordPiece, `[CLS]`/`[SEP]`, truncation, padding, attention masks, and a hidden-defaults gotcha |
| Step 2 | How ONNX works: what's inside a `.onnx` file, graph optimization, execution providers, contextual vectors |
| Step 3 | Mean pooling, and why ignoring padding matters (with a demo of the bug) |
| Step 4 | L2 normalization, and why it makes dot product = cosine similarity |
| Step 5 | Verifying the result against sentence-transformers |
| Step 6 | Packaging it as a drop-in `OnnxEmbedder` class |
| Step 7 | int8 quantization, and why you should always benchmark |
| Step 8 | Embedding ~1,500 FAQ documents, vector search from scratch with NumPy, and the step from search to RAG |
| Step 9 | Swapping it into real code, exporting your own models, a checklist, and what you gain in deployment |

## Quickstart

```bash
git clone https://github.com/Sanoy24/onnx-embeddings-from-scratch.git
cd onnx-embeddings-from-scratch
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
pip install sentence-transformers                     # optional: enables the "did we match?" checks
jupyter notebook onnx_embeddings_from_scratch.ipynb
```

Runs on a normal CPU. Python 3.10+.

The first run downloads the model (~90 MB fp32, plus ~23 MB for the int8 variant) from Hugging Face, and Step 8
downloads the FAQ dataset. Embedding the documents takes about a minute on a laptop.

## Credits

- Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0), which ships the ONNX export used here.
- Data: the public course FAQ of [DataTalks.Club](https://datatalks.club), downloaded at runtime from
  `datatalksclub.github.io/faq`. It is not redistributed in this repo.

## License

[MIT](LICENSE)
