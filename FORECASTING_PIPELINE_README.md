# Neural Forecasting Pipeline - Modular Architecture

This repository contains a refactored, modular neural forecasting pipeline using NeuralForecast with Ray Tune for hyperparameter optimization.

## 📁 Project Structure

```
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── forecasting_config.py          # Configuration settings and constants
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_preparation.py            # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py                # Load best model configurations
│   │   └── model_training.py              # Final model training
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── hyperparameter_tuning.py       # HPO with AutoModels
│   │   ├── cross_validation.py            # Cross-validation pipeline
│   │   └── prediction_evaluation.py       # Prediction and evaluation
│   └── utils/
│       ├── __init__.py
│       └── forecasting_utils.py           # Utility functions
├── forecasting_pipeline_main.py           # Main orchestration script
├── nf_ray.py                             # Original monolithic script (for reference)
└── model_definition.py                   # AutoModel definitions
```

## 🚀 Quick Start

### 1. Run Complete Pipeline
```bash
# Run the complete pipeline including hyperparameter optimization
python forecasting_pipeline_main.py

# Skip hyperparameter optimization and use existing configurations
python forecasting_pipeline_main.py --skip-hpo
```

### 2. Run Individual Components

#### Data Preparation Only
```bash
cd src/data
python data_preparation.py
```

#### Hyperparameter Tuning Only
```bash
cd src/pipeline
python hyperparameter_tuning.py
```

#### Cross-Validation Only
```bash
cd src/pipeline
python cross_validation.py
```

#### Model Training Only
```bash
cd src/models
python model_training.py
```

#### Prediction and Evaluation Only
```bash
cd src/pipeline
python prediction_evaluation.py
```

## 📋 Pipeline Steps

### Step 1: Environment Setup
- Initializes Ray for distributed computing
- Sets random seeds for reproducibility
- Configures logging and PyTorch settings

### Step 2: Data Preparation
- Loads dataset from `data/final/dataset.parquet`
- Renames columns to NeuralForecast format (`ds`, `y`, `unique_id`)
- Splits data into development and holdout test sets
- Identifies historical exogenous features

### Step 3: Hyperparameter Optimization (Optional)
- Uses AutoModels (AutoNHITS, etc.) with Ray Tune
- Saves best configurations to `tuning_results/best_hyperparameters.csv`
- Can be skipped if configurations already exist

### Step 4: Cross-Validation
- Loads best hyperparameters from CSV
- Runs time series cross-validation
- Evaluates model performance with MSE, MAE, RMSE

### Step 5: Final Model Training
- Trains final model on entire development set
- Uses best hyperparameters from tuning

### Step 6: Prediction and Evaluation
- Makes predictions on holdout test set
- Evaluates final performance metrics

## 🔧 Configuration

All configuration settings are centralized in `src/config/forecasting_config.py`:

### Key Parameters
- `FORECAST_HORIZON`: Number of steps to forecast (default: 7)
- `TEST_LENGTH_MULTIPLIER`: Test set size multiplier (default: 2)
- `CV_N_WINDOWS`: Number of cross-validation folds (default: 5)
- `NUM_SAMPLES_PER_MODEL`: HPO trials per model (default: 1)

### Data Configuration
- `DATA_PATH`: Path to input dataset
- `DATE_COLUMN`, `TARGET_COLUMN`: Column names to rename
- `UNIQUE_ID_VALUE`: Identifier for time series

### Model Configuration
- `FREQUENCY`: Time series frequency ('D' for daily)
- `LOCAL_SCALER_TYPE`: Scaling method ('standard')
- `LOSS_MAP`: Mapping of loss function names to objects

## 🧩 Modular Components

### Configuration Module (`src/config/`)
- **forecasting_config.py**: Centralized configuration management
- Easy to modify parameters without changing code
- Clear separation of settings from logic

### Data Module (`src/data/`)
- **data_preparation.py**: Data loading, cleaning, and splitting
- Handles column renaming and datetime conversion
- Validates data integrity before processing

### Models Module (`src/models/`)
- **model_loader.py**: Loads best hyperparameters from CSV
- **model_training.py**: Handles final model training
- Supports JSON parsing of complex hyperparameters

### Pipeline Module (`src/pipeline/`)
- **hyperparameter_tuning.py**: AutoModel HPO with Ray Tune
- **cross_validation.py**: Time series cross-validation
- **prediction_evaluation.py**: Final prediction and evaluation

### Utilities Module (`src/utils/`)
- **forecasting_utils.py**: Common utility functions
- Environment setup, seeding, metric calculations
- Reusable across different pipeline components

## 🎯 Benefits of Modular Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Easy to locate and modify specific functionality
- Reduced code duplication

### 2. **Debuggability**
- Each module can be tested independently
- Easier to isolate and fix issues
- Clear error propagation

### 3. **Extensibility**
- Easy to add new models or evaluation metrics
- Simple to modify individual pipeline steps
- Supports different data sources and formats

### 4. **Reusability**
- Components can be used in different projects
- Utility functions available across modules
- Configuration system adaptable to new scenarios

### 5. **Testing**
- Each module can have its own unit tests
- Integration testing at pipeline level
- Easier to validate individual components

## 📊 Output Files

- `tuning_results/best_hyperparameters.csv`: Best hyperparameters from HPO
- `lightning_logs/`: PyTorch Lightning training logs
- Console output with detailed progress and metrics

## 🔄 Migration from Original Script

The original `nf_ray.py` (528 lines) has been split into:

1. **Configuration**: 60 lines → `forecasting_config.py`
2. **Utilities**: 80 lines → `forecasting_utils.py`
3. **Data Processing**: 60 lines → `data_preparation.py`
4. **HPO**: 120 lines → `hyperparameter_tuning.py`
5. **Model Loading**: 100 lines → `model_loader.py`
6. **Cross-Validation**: 70 lines → `cross_validation.py`
7. **Training**: 50 lines → `model_training.py`
8. **Evaluation**: 80 lines → `prediction_evaluation.py`
9. **Orchestration**: 120 lines → `forecasting_pipeline_main.py`

## 🛠️ Customization

### Adding New Models
1. Update `model_definition.py` with new AutoModel
2. Add model creation logic to `model_loader.py`
3. Update configuration if needed

### Changing Loss Functions
1. Add new loss to `LOSS_MAP` in `forecasting_config.py`
2. No code changes needed elsewhere

### Modifying Evaluation Metrics
1. Update `calculate_evaluation_metrics()` in `forecasting_utils.py`
2. Add new metrics to evaluation output

### Custom Data Sources
1. Modify `load_and_prepare_data()` in `data_preparation.py`
2. Update column names in configuration

## 📝 Notes

- Ensure `model_definition.py` exists with `get_auto_models()` function
- Ray cluster must be available for distributed hyperparameter tuning
- GPU support is automatic if CUDA is available
- All relative paths assume execution from project root 