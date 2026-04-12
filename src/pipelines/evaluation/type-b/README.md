- eval_metrics_b.py : shared methods 
- manual: 
1. run total_eval_pipeline_b.py
2. run total_eval_normed_pipeline_b.py
3. run plot_eval_comparison_b.py 


pipline structure: 
  1. run_evals_stage1_b      
  — evaluate checkpoints, save per-sample predictions and test_results.csv to metrics/type-b/prediction/
  2. plot_eval_aggregate_b    
  — generate aggregate cross-run figures from prediction CSVs
  3. final_analysis   
   — compute composite ranking, save final_ranking.csv
                                 and final_per_digit.csv to metrics/type-b/prediction/ 