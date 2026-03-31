
---

## Project Structure

### Directory Layout

```
uob-ds-intro-to-ai-final-cw-2026/
│
├── src/                                ← all source code
│   ├── PROJECT_STRUCTURE.md            ← this file
│   │
│   ├── data/                           ← generated data
│   │   ├── images/                     ← generated images (gitignored)
│   │   │   └── type-b/                 ← MNIST-based images (.png files, gitignored)
│   │   └── type-b/                     ← type-b data files
│   │       ├── sentences_b.csv         ← generated sentences for type-b
│   │       ├── image_map_b.csv         ← sentence-to-image mapping
│   │       └── mnist_raw/              ← raw MNIST dataset (gitignored)
│   │
│   ├── data_generation/                ← Phase 1: dataset generation
│   │   ├── README.md
│   │   ├── type-a/                     ← shape dataset
│   │   │   ├── README.md
│   │   │   ├── generate_images_a.py    ← image generator for shapes (PIL)
│   │   │   ├── generate_sentences_a.py ← sentence generator for shapes
│   │   │   └── type-a-dataset/         ← generated .eps files
│   │   ├── type-b/                     ← MNIST number dataset
│   │   │   ├── README.md
│   │   │   ├── generate_images_b.py    ← image generator using MNIST
│   │   │   ├── generate_sentences_b.py ← sentence generator for numbers
│   │   │   └── analyse_distribution_b.py ← distribution analysis script
│   │   └── type-c/                     ← tic-tac-toe dataset
│   │       ├── README.md
│   │       ├── type_c_core.py          ← core logic for type-c
│   │       ├── type_c_image_generator.py ← image generator for tic-tac-toe
│   │       └── type_c_sentence_generator.py ← sentence generator for tic-tac-toe
│   │
│   ├── embeddings/                     ← Phase 2: text embeddings
│   │   ├── README.md
│   │   ├── tfidf_embeddings.py         ← TF-IDF embeddings
│   │   ├── bert_embeddings.py          ← BERT embeddings
│   │   ├── sbert_embeddings.py         ← SBERT embeddings
│   │   ├── glove_embeddings.py         ← GloVe embeddings
│   │   ├── word2vec_embeddings.py      ← Word2Vec embeddings
│   │   ├── one_hot_coding_embeddings.py      ← One-Hot Coding embeddings 
│   │   └── sentence_transformer_embeddings.py ← Sentence Transformer embeddings
│   │
│   ├── preprocessing/                  ← image preprocessing utilities
│   │   └── image_transforms.py
│   │
│   ├── models/                         ← Phase 2: model definitions
│   │   └── README.md
│   │
│   ├── training/                       ← Phase 2: training loop
│   │   └── README.md
│   │
│   └── evaluation/                     ← Phase 2: evaluation & analysis
│       └── README.md
│
├── notebooks/                          ← shared notebooks
│   ├── README.md
│   └── dataset_generation.ipynb
│
├── results/                            ← saved experiment outputs
│   ├── README.md
│   ├── figures/                        ← plots for the report
│   │   └── README.md
│   └── metrics/                        ← accuracy scores, loss curves (.json)
│       └── README.md
│
│
├── .gitignore
├── requirements.txt
├── README.md
└── PREPROCESSING_STRATEGY.md
```
