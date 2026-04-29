
---

## Project Structure

- The project structure have main directories, but for each dataset team's convenience, we did not force to follow the first structure defined in the first meeting. (Regard to this file to find experiment results for type-a,b,c respectively)

```
uob-ds-intro-to-ai-final-cw-2026/
│
├── src/                                        ← all source code
│   │
│   ├── LLM/                                    ← commercial LLM evaluation (Type-A)
│   │   ├── gemini.py                           ← Gemini API wrapper
│   │   ├── run_gemini_type_a.py                ← run Gemini on Type-A test set
│   │   ├── prepare_type_a_test_set.py
│   │   ├── evaluate_type_a_llm_full_metrics.py
│   │   ├── evaluate_type_a_llm_retrieval.py
│   │   ├── evaluate_type_a_llm_semantic.py
│   │   ├── type_a_test_set.csv
│   │   ├── type_a_llm_outputs_full.csv
│   │   ├── type_a_llm_metrics_summary.csv
│   │   ├── type_a_llm_metrics_details.csv
│   │   ├── type_a_llm_retrieval_summary.csv
│   │   └── type_a_llm_retrieval_details.csv
│   │
│   ├── config/                                 ← experiment configuration (JSON + Python)
│   │   ├── __init__.py
│   │   ├── cnn_layers.json                     ← per-dataset CNN layer architecture configs
│   │   ├── embeddings.json                     ← embedding method configs and output dims
│   │   ├── hyperparams.json                    ← hyperparameter tuning grids
│   │   ├── loss.json                           ← loss function configs per model/dataset
│   │   ├── training.json                       ← batch size, epochs, LR, scheduler
│   │   └── paths.py                            ← centralised path resolution
│   │
│   ├── data/                                   ← generated data CSVs + raw assets
│   │   ├── images/                             ← image directories (gitignored)
│   │   │   └── type-b/                         ← 10,008 Type-B images (b_0.png … b_10007.png)
│   │   ├── type-a/
│   │   │   ├── master.csv
│   │   │   ├── sentences_a.csv
│   │   │   └── dataloader.py
│   │   ├── type-b/
│   │   │   ├── sentences_b.csv                 ← 10,008 sentences for MNIST dataset
│   │   │   ├── image_map_b.csv                 ← sentence ↔ image filename mapping
│   │   │   ├── mnist_raw/                      ← raw MNIST binary files (train + test)
│   │   │   └── splits/
│   │   │       └── type_b_splits_seed42.csv    ← train/val/test split manifest (seed=42)
│   │   └── type-c/
│   │       ├── sentences_c.csv
│   │       ├── image_map_c.csv
│   │       └── type_c_dataset.json
│   │
│   ├── data_generation/                        ← Phase 1: dataset generation scripts
│   │   ├── README.md
│   │   ├── type-a/                             ← shape + colour description dataset
│   │   │   ├── generate_images_a.py
│   │   │   ├── generate_sentences_a.py
│   │   │   ├── dataset_generator_a_20k.py
│   │   │   ├── dataset_generator_a_20k_PIL.py
│   │   │   ├── relation_shapes_generator_a.py
│   │   │   └── converter.py
│   │   ├── type-b/                             ← coloured MNIST digit dataset
│   │   │   ├── generate_images_b.py
│   │   │   ├── generate_sentences_b.py
│   │   │   └── analyse_distribution_b.py
│   │   └── type-c/                             ← tic-tac-toe board dataset
│   │       ├── type_c_core.py
│   │       ├── type_c_image_generator.py
│   │       ├── type_c_sentence_generator.py
│   │       └── task5_tictactoe.py
│   │
│   ├── embeddings/                             ← Phase 2: text embedding implementations
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── TypeC_TF-IDF+OneHot.py             ← Type-C specific TF-IDF + one-hot script
│   │   ├── non-pretrained/                     ← corpus-trained (no external weights)
│   │   │   ├── tfidf_embeddings.py
│   │   │   ├── tfidf_lsa_embeddings.py         ← TF-IDF + TruncatedSVD
│   │   │   ├── tfidf_weighted_word2vec_embeddings.py
│   │   │   ├── word2vec_skipgram_embeddings.py
│   │   │   └── one_hot_encoding.py
│   │   ├── pretrained/                         ← pretrained model embeddings
│   │   │   ├── bert_mean_embeddings.py         ← generic (all datasets)
│   │   │   ├── bert_pooler_embeddings.py
│   │   │   ├── tinybert_mean_embeddings.py
│   │   │   ├── tinybert_pooler_embeddings.py
│   │   │   ├── sbert_embeddings.py
│   │   │   ├── glove_embedding.py
│   │   │   ├── pretrained_word2vec_embeddings.py
│   │   │   ├── only_type_a_B_mean.py           ← dataset-specific variants
│   │   │   ├── only_type_a_B_pooler.py
│   │   │   ├── only_type_a_TB_mean.py
│   │   │   ├── only_type_a_TB_pooler.py
│   │   │   ├── only_type_a_p_w2v.py
│   │   │   ├── only_type_a_pretrained_word2vec_embeddings.py
│   │   │   ├── only_type_a_sbert_embeddings.py
│   │   │   ├── only_type_b_bert_mean_embeddings.py
│   │   │   ├── only_type_b_bert_pooler_embeddings.py
│   │   │   ├── only_type_b_tinybert_mean_embeddings.py
│   │   │   ├── only_type_b_tinybert_pooler_embeddings.py
│   │   │   ├── only_type_c_glove_embedding.py
│   │   │   └── fine-tune/
│   │   │       └── only_typeb_finetune_sbert.py ← SBERT fine-tuning on Type-B corpus
│   │   └── computed-embeddings/                ← pre-computation scripts + .pt outputs
│   │       ├── type-a/
│   │       │   └── add_emb_to_master.py
│   │       ├── type-b/
│   │       │   ├── generate_embeddings_type_b.py
│   │       │   ├── inspect_embeddings.py
│   │       │   ├── tfidf_lsa_variance_analysis.py
│   │       │   ├── tfidf_lsa_variance_type_b.csv ← explained variance output
│   │       │   └── results/                    ← .pt embedding files (stored on Drive)
│   │       │       ├── tfidf_lsa_embedding_result_typeb.pt
│   │       │       ├── tfidf_w2v_embedding_result_typeb.pt
│   │       │       ├── word2vec_skipgram_embedding_result_typeb.pt
│   │       │       ├── word2vec_pretrained_embedding_result_typeb.pt
│   │       │       ├── glove_embedding_result_typeb.pt
│   │       │       ├── sbert_embedding_result_typeb.pt
│   │       │       ├── sbert_finetuned_embedding_result_typeb.pt
│   │       │       ├── bert_mean_embedding_result_typeb.pt
│   │       │       ├── bert_pooler_embedding_result_typeb.pt
│   │       │       ├── tinybert_mean_embedding_result_typeb.pt
│   │       │       └── tinybert_pooler_embedding_result_typeb.pt
│   │       └── type-c/
│   │           └── README.md
│   │
│   ├── models/                                 ← CNN model definitions (image encoder)
│   │   ├── README.md
│   │   ├── CNN.py                              ← simple 1-conv-layer baseline
│   │   ├── CNN2.py
│   │   ├── CNN2Layer_DSA.py
│   │   ├── googleNet.py
│   │   ├── resnet18.py
│   │   ├── resnet_dsa.py
│   │   └── type-b/                             ← Type-B specific model variants
│   │       ├── cnn_1layer.py                   ← baseline CNN (Stage 1)
│   │       ├── cnn_2layer.py
│   │       ├── cnn_3layer.py                   ← deeper CNN (Stage 2 analytic axis)
│   │       ├── alexnet.py
│   │       └── resnet18_pt.py                  ← pretrained ResNet-18 (Stage 2)
│   │
│   ├── pipelines/                              ← end-to-end experiment runners
│   │   ├── data_loaders/                       ← PyTorch Dataset / DataLoader classes
│   │   │   ├── __init__.py
│   │   │   ├── type_a_dataloader.py
│   │   │   ├── type_b_loader.py                ← loads pre-saved .pt embeddings
│   │   │   └── one_emb.py
│   │   │
│   │   ├── training/                           ← training loop + utilities
│   │   │   ├── cosine_loss.py                  ← cosine embedding loss wrapper
│   │   │   ├── earlyStopping.py
│   │   │   ├── type-a/
│   │   │   │   ├── train.py
│   │   │   │   └── training.py
│   │   │   ├── type-b/
│   │   │   │   ├── shared.py                   ← shared EMBEDDING_CONFIGS + MODEL_CONFIGS
│   │   │   │   ├── train_type_b.py             ← Stage 1: sweep all embeddings × cnn_1layer
│   │   │   │   ├── train_type_b_stage2.py      ← Stage 2: best emb × multiple architectures
│   │   │   │   └── plot_training_curves_b.py
│   │   │   └── type-c/                         ← placeholder (empty)
│   │   │
│   │   ├── evaluation/                         ← retrieval evaluation scripts
│   │   │   ├── README.md
│   │   │   ├── type-a/
│   │   │   │   ├── evaluate_type_a_run.py
│   │   │   │   ├── build_type_a_leaderboard.py
│   │   │   │   ├── compare_type_a_runs.py
│   │   │   │   ├── rerun_all_type_a_evals.py
│   │   │   │   ├── type_a_label_parser.py
│   │   │   │   └── type_a_metrics.py
│   │   │   ├── type-b/
│   │   │   │   ├── README.md
│   │   │   │   ├── eval_metrics_b.py           ← top-k, MRR, mean rank, cosine sim
│   │   │   │   ├── run_evals_stage1_b.py       ← evaluate Stage 1 (non-normalised)
│   │   │   │   ├── run_evals_stage1_normed_b.py ← evaluate Stage 1 (normalised)
│   │   │   │   ├── run_evals_stage2_b.py       ← evaluate Stage 2 models
│   │   │   │   ├── total_eval_pipeline_b.py    ← full eval pipeline (non-normalised)
│   │   │   │   ├── total_eval_pipeline_normed_b.py
│   │   │   │   ├── total_eval_pipeline_all_b.py ← combined Stage 1 + 2 + LLM
│   │   │   │   ├── plot_eval_aggregate_b.py
│   │   │   │   ├── plot_eval_comparison_b.py
│   │   │   │   ├── final_analysis.py
│   │   │   │   ├── final_analysis_normed.py
│   │   │   │   ├── final_analysis_combined.py
│   │   │   │   └── openrouter_comparison.py    ← Gemini-Lite vs CNN comparison
│   │   │   └── type-c/                         ← placeholder (empty)
│   │   │
│   │   └── results/                            ← all experiment outputs
│   │       ├── README.md
│   │       ├── checkpoints/                    ← saved .pt model weights
│   │       │   ├── sbert_finetuned_typeb/      ← fine-tuned SBERT weights (HF format)
│   │       │   ├── type-a/
│   │       │   ├── type-b/
│   │       │   │   ├── *.pt                    ← Stage 1 best checkpoints
│   │       │   │   ├── normalised/             ← Stage 1 normalised checkpoints
│   │       │   │   └── s2/
│   │       │   │       ├── normalised/         ← Stage 2 normalised checkpoints
│   │       │   │       └── non-normalised/
│   │       │   └── type-c/
│   │       ├── metrics/                        ← CSV experiment result files
│   │       │   ├── type-a/                     ← per-run summary + details CSVs
│   │       │   └── type-b/
│   │       │       ├── non-normalised/         ← Stage 1 training logs
│   │       │       ├── normalised/             ← Stage 1 normalised training logs
│   │       │       ├── s2-non-normalised/      ← Stage 2 training logs
│   │       │       ├── s2-normalised/
│   │       │       ├── prediction/             ← test set predictions (Stage 1)
│   │       │       ├── prediction-normalised/  ← leaderboard + ranking CSVs
│   │       │       ├── prediction-s2/          ← Stage 2 predictions
│   │       │       ├── prediction-s2-normalised/
│   │       │       ├── prediction-commercial-ai/ ← Gemini-Lite LLM predictions
│   │       │       └── prediction-combined/    ← merged Stage 1 + Stage 2 + LLM
│   │       └── figures/                        ← all plots
│   │           └── type-b/
│   │               ├── train/                  ← loss curves, val metric curves
│   │               │   ├── comparison/         ← normalised vs non-normalised
│   │               │   └── normalised/
│   │               └── evaluation/             ← retrieval metric plots
│   │                   ├── llm/
│   │                   ├── normalised/
│   │                   ├── comparison/
│   │                   ├── combined/
│   │                   └── s2/
│   │                       ├── normalised/
│   │                       └── non-normalised/
│   │
│   └── result/                                 ← Type-C results (Zhenmao / Gia)
│       └── type-C/
│           ├── TFIDF/cnn/  TFIDF/resnet/
│           ├── glove/cnn/  glove/resnet/
│           ├── sbert/cnn/  sbert/resnet/
│           ├── _plots/                         ← aggregated Type-C evaluation plots
│           ├── clip_type_c_eval.py             ← CLIP zero-shot evaluation
│           ├── openrouter_typec_fullrun_best.py ← OpenRouter LLM evaluation
│           ├── openrouter_typec_sbert_detailed_google-gemini-2.0-flash-lite-001/
│           ├── test_consistency.py
│           ├── by_moves_results.csv
│           ├── failures.csv
│           └── test_results.csv
│
├── notebooks/                                  ← Colab training / generation notebooks
│   ├── README.md
│   ├── dataset_generation.ipynb
│   └── train-evaluation/
│       ├── type-a/
│       │   └── full_pipeline_DSA.ipynb
│       ├── type-b/
│       │   ├── _setup.py                       ← Colab environment bootstrap
│       │   ├── _push_results.py                ← push results back to Drive/repo
│       │   ├── colab_train_b.ipynb             ← Stage 1 (non-normalised sweep)
│       │   ├── colab_train_b_normalised.ipynb  ← Stage 1 (L2-normalised sweep)
│       │   └── colab_train_b_stage2.ipynb      ← Stage 2 (architecture comparison)
│       └── type-c/                             ← placeholder (empty)
│
├── report/                                     ← LaTeX report
│   ├── README.md
│   ├── formatting.md
│   ├── anthology.bib.txt
│   ├── latex/
│   │   ├── acl_latex.tex                       ← main report source
│   │   ├── acl.sty
│   │   ├── acl_natbib.bst
│   │   └── custom.bib
│   └── figures/
│       ├── type_a/
│       ├── type_b/
│       └── type_c/
│
├── .gitmodules
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```
