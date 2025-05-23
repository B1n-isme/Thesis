"""
Main orchestration script for the neural forecasting pipeline.

This script coordinates the entire forecasting workflow:
1. Environment setup
2. Data preparation
3. Hyperparameter tuning (optional)
4. Cross-validation
5. Final model training
6. Prediction and evaluation

Usage:
    python forecasting_pipeline_main.py [--skip-hpo]
"""
import argparse
from datetime import datetime

from src.utils.forecasting_utils import setup_environment
from src.data.data_preparation import prepare_forecasting_data
from src.pipeline.hyperparameter_tuning import run_complete_hpo_pipeline
from src.pipeline.cross_validation import run_complete_cv_pipeline
from src.models.model_training import create_and_train_final_model
from src.pipeline.prediction_evaluation import run_prediction_evaluation
from src.config.forecasting_config import SEED, RAY_ADDRESS, RAY_NUM_CPUS, RAY_NUM_GPUS


def main(skip_hpo=False):
    """Run the complete neural forecasting pipeline."""
    
    # 1. Setup Environment
    print("=" * 50)
    print("NEURAL FORECASTING PIPELINE")
    print("=" * 50)
    
    ray_config = {
        'address': RAY_ADDRESS,
        'num_cpus': RAY_NUM_CPUS,
        'num_gpus': RAY_NUM_GPUS
    }
    setup_environment(seed=SEED, ray_config=ray_config)
    
    # 2. Data Preparation
    print("\n" + "=" * 50)
    print("STEP 1: DATA PREPARATION")
    print("=" * 50)
    
    df, df_development, df_final_holdout_test, hist_exog_list = prepare_forecasting_data()
    print(f"Historical exogenous features: {len(hist_exog_list)} features")
    
    # 3. Hyperparameter Tuning (Optional)
    if not skip_hpo:
        print("\n" + "=" * 50)
        print("STEP 2: HYPERPARAMETER OPTIMIZATION")
        print("=" * 50)
        
        nf_hpo, all_best_configs, csv_filename = run_complete_hpo_pipeline(df_development)
        if csv_filename:
            print(f"HPO completed. Best configurations saved to: {csv_filename}")
        else:
            print("HPO failed. Proceeding with default configurations.")
    else:
        print("\n" + "=" * 50)
        print("STEP 2: HYPERPARAMETER OPTIMIZATION (SKIPPED)")
        print("=" * 50)
        print("Using existing hyperparameter configurations...")
    
    # 4. Cross-Validation
    print("\n" + "=" * 50)
    print("STEP 3: CROSS-VALIDATION")
    print("=" * 50)
    
    cv_results_df, cv_metrics, nf_cv = run_complete_cv_pipeline(df_development)
    
    if cv_results_df is not None:
        print(f"Cross-validation completed successfully!")
        print(f"CV results shape: {cv_results_df.shape}")
    else:
        print("Cross-validation failed. Check hyperparameter file.")
        return
    
    # 5. Final Model Training
    print("\n" + "=" * 50)
    print("STEP 4: FINAL MODEL TRAINING")
    print("=" * 50)
    
    nf_final_train, model_instance, best_params, final_loss_object = create_and_train_final_model(df_development)
    
    if nf_final_train is None:
        print("Final model training failed. Aborting pipeline.")
        return
    
    # 6. Prediction and Evaluation
    print("\n" + "=" * 50)
    print("STEP 5: PREDICTION AND EVALUATION")
    print("=" * 50)
    
    predictions_on_test, evaluation_results = run_prediction_evaluation(
        nf_final_train, 
        df_final_holdout_test, 
        model_name='NHITS'
    )
    
    # 7. Final Summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    
    if evaluation_results:
        print(f"Model: {evaluation_results['model_name']}")
        print(f"Test MAE: {evaluation_results['test_mae']:.4f}")
        print(f"Test RMSE: {evaluation_results['test_rmse']:.4f}")
        print(f"Evaluation DataFrame shape: {evaluation_results['evaluation_df'].shape}")
    else:
        print("Evaluation failed - no results available.")
    
    print(f"\nPipeline execution finished at: {datetime.now()} (Ho Chi Minh City Time)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Forecasting Pipeline")
    parser.add_argument(
        '--skip-hpo', 
        action='store_true', 
        help='Skip hyperparameter optimization and use existing configurations'
    )
    
    args = parser.parse_args()
    
    try:
        main(skip_hpo=args.skip_hpo)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise 