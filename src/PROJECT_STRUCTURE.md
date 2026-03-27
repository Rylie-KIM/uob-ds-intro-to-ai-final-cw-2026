
---

## Project Structure

### Directory Layout

```
uob-ds-intro-to-ai-final-cw-2026/
│
├── data/                               ← generated data (gitignored — stored on Google Drive)
│   ├── type1a/                         ← shape images (.png files)
│   ├── type1b/                         ← MNIST number images (.png files)
│   ├── type1c/                         ← tic-tac-toe images (.png files)
│   └── splits/                         ← train/val/test split indices (.json files)
│
├── src/                                ← all source code
│   ├── data_generation/                ← Phase 1: owned by all 5 members
│   │   ├── generate_sentences_1a.py    ← sentence generator for shapes (Person 1)
│   │   ├── generate_sentences_1b.py    ← sentence generator for numbers (Person 3)
│   │   ├── generate_sentences_1c.py    ← sentence generator for tic-tac-toe (Person 4)
│   │   ├── generate_images_1a.py       ← image generator for shapes using PIL (Person 2)
│   │   ├── generate_images_1b.py       ← image generator for numbers using MNIST (Person 3)
│   │   ├── generate_images_1c.py       ← image generator for tic-tac-toe using PIL (Person 4)
│   │   └── make_splits.py              ← train/val/test split script (Person 5)
│   │
│   ├── embeddings/                     ← Phase 2: text representation (Person 1)
│   │   ├── tfidf_embeddings.py
        - bert_embeddings.py
│   │   ├── glove_embeddings.py
│   │   └── sentence_transformer_embeddings.py
│   │
│   ├── models/                         ← Phase 2: model definitions (Person 2, 4)
│   │   ├── baseline_cnn.py
│   │   └── mlp_baseline.py
│   │
│   ├── training/                       ← Phase 2: training loop (Person 2, 3)
│   │   └── train.py
│   │
│   └── evaluation/                     ← Phase 2: evaluation & analysis (Person 5)
│       ├── evaluate.py
│       ├── llm_comparison.py
│       └── error_analysis.py
│
├── notebooks/                          ← shared Colab notebooks
│   ├── 01_data_exploration.ipynb       ← EDA, example images, sentence distribution plots
│   ├── 02_baseline_training.ipynb      ← baseline CNN training and results
│   └── 03_experiments.ipynb            ← analytic axis experiments
│
├── results/                            ← saved experiment outputs (committed to git)
│   ├── figures/                        ← plots for the report
│   └── metrics/                        ← accuracy scores, loss curves (.json)
│
├── .gitignore
├── requirements.txt
├── README.md
├── REPORT_DRAFT.md
└── PLAN.md
```