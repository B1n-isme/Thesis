"""
Prediction and evaluation module for neural forecasting models.
"""
import pandas as pd
from datetime import datetime


def make_predictions(nf_final_train, df_final_holdout_test):
    """Make predictions on the final holdout test set."""
    print("\nMaking predictions on the final holdout test set...")
    
    # NeuralForecast's predict method can take df_final_holdout_test directly.
    # It will use the historical part of each series in df_final_holdout_test
    # to generate the initial input window, and then predict 'h' steps.
    # 
    # The predict() method forecasts h steps from the last timestamp in the training data for each unique_id.
    # We need to ensure these forecasted ds values match our df_final_holdout_test.
    
    predictions_on_test = nf_final_train.predict(df=df_final_holdout_test)
    
    print(f"Predictions shape: {predictions_on_test.shape}")
    print(f"Predictions columns: {predictions_on_test.columns.tolist()}")
    
    return predictions_on_test


def evaluate_predictions(df_final_holdout_test, predictions_on_test, model_name='NHITS'):
    """Evaluate predictions against actual values."""
    # For this example, assuming test_length was set up to align with forecasting h steps.
    # If predict() output doesn't perfectly align or you need more control, consider predict(futr_df=...).
    # Let's merge based on 'unique_id' and 'ds'.
    
    # final_evaluation_df = pd.merge(
    #     df_final_holdout_test,
    #     predictions_on_test,
    #     on=['unique_id', 'ds'],
    #     how='left'  # Use left to keep all test points; predictions might be shorter if h < test_length
    # )
    # final_evaluation_df.dropna(inplace=True)  # If some predictions couldn't be made or aligned.
    
    # print(f"Final evaluation dataframe columns: {final_evaluation_df.columns.tolist()}")
    # print(f"Final evaluation dataframe shape: {final_evaluation_df.shape}")
    
    # if final_evaluation_df.empty:
    #     print("Warning: No aligned predictions found for evaluation.")
    #     return None
    
    # Calculate evaluation metrics
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mse, mae, rmse
    
    # test_actuals = final_evaluation_df['y']
    # test_preds = final_evaluation_df[model_name]
    
    # final_mae = mae(test_actuals, test_preds)
    # final_rmse = rmse(test_actuals, test_preds)
    
    # print(f"\nFinal Evaluation on Holdout Test Set for {model_name}:")
    # print(f"  Test MAE: {final_mae:.4f}")
    # print(f"  Test RMSE: {final_rmse:.4f}")
    
    # return {
    #     'model_name': model_name,
    #     'test_mae': final_mae,
    #     'test_rmse': final_rmse,
    #     'evaluation_df': final_evaluation_df
    # }

    evaluation_df = evaluate(predictions_on_test.drop(columns='cutoff'), metrics=[mse, mae, rmse])
    evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
    return evaluation_df


def run_prediction_evaluation(nf_final_train, df_final_holdout_test, model_name='NHITS'):
    """Run the complete prediction and evaluation pipeline."""
    # Make predictions
    predictions = make_predictions(nf_final_train, df_final_holdout_test)
    
    # Evaluate predictions
    evaluation_results = evaluate_predictions(df_final_holdout_test, predictions, model_name)
    
    return predictions, evaluation_results


if __name__ == "__main__":
    from src.data.data_preparation import prepare_forecasting_data
    from src.models.model_training import create_and_train_final_model
    
    # Prepare data
    _, df_development, df_final_holdout_test, _ = prepare_forecasting_data()
    
    # Create and train final model
    nf_final, model, params, loss = create_and_train_final_model(df_development)
    
    if nf_final is not None:
        # Run prediction and evaluation
        predictions, eval_results = run_prediction_evaluation(nf_final, df_final_holdout_test)
        
        if eval_results is not None:
            print(f"\nPrediction and evaluation completed successfully!")
            # print(f"Test MAE: {eval_results['test_mae']:.4f}")
            # print(f"Test RMSE: {eval_results['test_rmse']:.4f}")
            print(f'Evaluation results: {eval_results}')
        else:
            print("Evaluation failed - no aligned predictions found.")
    else:
        print("Cannot run prediction - model training failed.")
    
    print(f"\nPipeline execution finished at: {datetime.now()} (Ho Chi Minh City Time)") 