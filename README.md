# Recurrent Neural Network: (RNN with Bahdanau Attention)

A production-grade, modular implementation of a Sequence-to-Sequence (Seq2Seq) conversational model built from scratch in PyTorch. This architecture features a Bidirectional LSTM Encoder, an additive Bahdanau Attention mechanism, an LSTM Decoder, and SentencePiece tokenization. The training and inference pipelines are optimized for modern workflows using the `uv` package manager, Automatic Mixed Precision (AMP), gradient accumulation, dynamic dataset sharding, and beam search decoding.

---

## **System Architecture & Components**

The model is partitioned into distinct modular components housed within the source tree:

* **Bidirectional LSTM Encoder (`src/model/encoder.py`):** Encodes input tokens using an embedding layer, applies dropout, and processes variable-length sequences via packed padded sequences (`pack_padded_sequence` and `pad_packed_sequence`). It merges bidirectional LSTM states by summing forward and backward projections across layers.

* **Bahdanau Attention (`src/model/attention.py`):** Computes additive alignment scores between the encoder's hidden outputs and the decoder's current hidden state, producing normalized attention weights with optional masking for padding tokens.

* **LSTM Decoder (`src/model/decoder.py`):** Combines target token embeddings with context vectors derived from the attention mechanism. It passes the combined representations through an LSTM layer and a linear projection layer to predict the next token logits over the vocabulary.

* **Seq2Seq Coordinator (`src/model/seq2seq.py`):** Wraps the encoder and decoder to execute the forward pass, managing teacher forcing ratios dynamically during training iterations.

---

## **Project Structure**

```text
├── checkpoints/
│   ├── best.pt                # Best model based on validation loss
│   └── latest.pt              # Latest epoch checkpoint
├── data/
│   ├── processed/
│   │   ├── train/             # Sharded training dataset tensors (.pt)
│   │   └── valid/             # Sharded validation dataset tensors (.pt)
│   ├── raw/                   # Raw data documentation and files
│   └── tokenizer/             # SentencePiece model and vocabulary files
├── logs/                      # TensorBoard event trace logs
├── src/
│   ├── model/
│   │   ├── attention.py       # Bahdanau attention implementation
│   │   ├── decoder.py         # Attention-based LSTM decoder
│   │   ├── encoder.py         # Bidirectional LSTM encoder
│   │   └── seq2seq.py         # Core sequence-to-sequence model wrapper
│   ├── testing/               # Component unit tests and inspection scripts
│   ├── utils/
│   │   ├── beam_search.py     # Beam search decoding utilities
│   │   ├── checkpoint.py      # State saving and loading handlers
│   │   ├── collate.py         # Dynamic batch padding collate function
│   │   ├── create_dirs.py     # Directory initialization helper
│   │   ├── dataset.py         # Shard dataset streaming class
│   │   └── multishards.py     # Shard generation/management utilities
│   ├── config.py              # Centralized hyperparameter configuration dataclass
│   ├── dataset.py             # Dataset processing pipelines
│   ├── inference.py           # CLI chat interface script with beam search
│   ├── train.py               # Training entry point, shard discovery, and resume logic
│   └── trainer.py             # Trainer class with AMP, gradient accumulation, and logging
├── .gitignore
├── Output.png                 # Sample CLI execution screenshot
├── pyproject.toml             # Project metadata and dependencies
└── uv.lock                    # Locked dependency versions

```

---

## **Hyperparameters & Configuration**

Centralized configurations are defined in `src/config.py` using a frozen dataclass:

* **Vocabulary Size:** 16,000 tokens (SentencePiece model)

* **Embedding Dimension:** 256

* **Hidden Sizes:** Encoder hidden size: 512, Decoder hidden size: 512 (Bidirectional encoder doubles effective hidden dimensions for attention and decoder inputs)

* **Network Depth:** 2 LSTM layers with 0.3 dropout

* **Training Hyperparameters:** Batch size of 28 with gradient accumulation steps set to 2, 10 epochs, AdamW optimizer ($\text{lr} = 3\times 10^{-4}$, weight decay $= 1\times 10^{-4}$), gradient clipping threshold at 1.0, and Cosine Annealing learning rate scheduling.

---

## **Data Pipeline & Sharding**

To handle large conversational datasets efficiently without overloading RAM, the dataset is serialized into compressed PyTorch tensor shards (`.pt`) inside `data/processed/train/` and `data/processed/valid/`.

* **Collation (`src/utils/collate.py`):** Automatically pads variable-length sequences to uniform lengths within a batch using predefined padding tokens (`cfg.PAD`).

* **Streaming:** The training loop instantiates `ShardDataset` sequentially per shard, garbage-collecting memory and clearing CUDA cache between shards to maintain optimal resource usage.

---

## **Training & Checkpoint Management**

The training script (`src/train.py`) features robust fault tolerance and continuity handling:

* **Checkpoint Detection:** Automatically scans the `checkpoints/` directory for `batch_latest.pt` (mid-epoch shard state) and `latest.pt` (epoch completion state). It intelligently determines whether to resume from a specific shard or advance to the next epoch.

* **Trainer Loop (`src/trainer.py`):** Integrates PyTorch's `GradScaler` for Automatic Mixed Precision (AMP), executes gradient accumulation, logs loss metrics to TensorBoard (`SummaryWriter`), and preserves models that yield optimal validation scores (`best.pt`).

---

## **Inference & Interactive CLI**

The chatbot interface (`src/inference.py`) loads the trained model checkpoint, initializes a SentencePiece processor, and runs a stateful conversational loop in the terminal.

* **Features:** Maintains a rolling conversation history buffer using a `deque` (up to `MAX_HISTORY = 8` turns) to preserve conversational context across prompts.

* **Decoding:** Employs beam search decoding with length penalties (`beam_search_decode`) instead of greedy sampling to generate fluent responses.

---

## **Getting Started**

### **1. Prerequisites & Installation**

Ensure you have `uv` installed. Clone the repository and sync the environment dependencies specified in `pyproject.toml`:

```bash
git clone [https://github.com/Shaahmir/Recurrent-Neural-Network.git](https://github.com/Shaahmir/Recurrent-Neural-Network.git)
cd Recurrent-Neural-Network
uv sync
```

### **2. Training the Model**

To start training the Seq2Seq model using the processed shards and configuration settings:

```bash
uv run python -m src.train
```

The trainer automatically handles checkpoint resumption, gradient accumulation, learning rate scheduling via Cosine Annealing, and logs metrics to TensorBoard. Monitor training progress using:

```bash
uv run tensorboard --logdir logs
```

### **3. Running Inference (Chatbot CLI)**

Interact with the trained chatbot using the beam search inference script:

```bash
cd src
uv run python -m inference
```

**Commands within the chat interface:**

* Type your message naturally and press `Enter` to chat with the AI.
* Type `/reset` to clear the conversation history buffer.
* Type `exit` to close the session.

---
