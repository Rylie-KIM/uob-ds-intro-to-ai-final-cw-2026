import subprocess
import sys

RUNS = [
    ("0_CNN2Layer_sbert_CosineLoss", "sbert_emb", "CNN2Layer", "sbert", "CosineLoss"),
    ("1_CNNPool_sbert_CosineLoss", "sbert_emb", "CNNPool", "sbert", "CosineLoss"),
    ("2_CNNStride_sbert_CosineLoss", "sbert_emb", "CNNStride", "sbert", "CosineLoss"),
    ("3_GoogleNet_sbert_CosineLoss", "sbert_emb", "GoogleNet", "sbert", "CosineLoss"),
    ("4_ResnetDSA_sbert_CosineLoss", "sbert_emb", "ResnetDSA", "sbert", "CosineLoss"),
    ("5_ResnetDSA_TB_pooler_CosineLoss", "TB_pooler_emb", "ResnetDSA", "TB_pooler", "CosineLoss"),
    ("6_ResnetDSA_TB_mean_CosineLoss", "TB_mean_emb", "ResnetDSA", "TB_mean", "CosineLoss"),
    ("7_ResnetDSA_B_pooler_CosineLoss", "B_pooler_emb", "ResnetDSA", "B_pooler", "CosineLoss"),
    ("8_ResnetDSA_B_mean_CosineLoss", "B_mean_emb", "ResnetDSA", "B_mean", "CosineLoss"),
    ("9_ResnetDSA_P_wvec_CosineLoss", "P_wvec", "ResnetDSA", "P_wvec", "CosineLoss"),
    ("10_ResnetDSA_sbert_MSELoss", "sbert_emb", "ResnetDSA", "sbert", "MSELoss"),
    ("11_ResnetDSA_TB_pooler_MSELoss", "TB_pooler_emb", "ResnetDSA", "TB_pooler", "MSELoss"),
    ("12_ResnetDSA_TB_mean_MSELoss", "TB_mean_emb", "ResnetDSA", "TB_mean", "MSELoss"),
    ("13_ResnetDSA_B_pooler_MSELoss", "B_pooler_emb", "ResnetDSA", "B_pooler", "MSELoss"),
    ("14_ResnetDSA_B_mean_MSELoss", "B_mean_emb", "ResnetDSA", "B_mean", "MSELoss"),
    ("15_ResnetDSA_P_wvec_MSELoss", "P_wvec", "ResnetDSA", "P_wvec", "MSELoss"),
]

for run_name, embedding_col, model_name, embedding_name, loss_name in RUNS:
    predictions_pt = f"src/pipelines/results/checkpoints/type-a/{run_name}_predictions.pt"

    cmd = [
        sys.executable,
        "src/pipelines/evaluation/type-a/evaluate_type_a_run.py",
        "--master_csv", "src/data/type-a/master.csv",
        "--predictions_pt", predictions_pt,
        "--embedding_col", embedding_col,
        "--run_name", run_name,
        "--model_name", model_name,
        "--embedding_name", embedding_name,
        "--loss_name", loss_name,
        "--test_indices_csv", "src/pipelines/results/checkpoints/type-a/test_indices.csv",
        "--output_dir", "src/pipelines/results/metrics/type-a",
    ]

    print(f"\nRunning: {run_name}")
    subprocess.run(cmd, check=True)

print("\nAll Type-A evaluations rerun successfully.")