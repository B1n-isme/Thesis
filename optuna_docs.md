# Optuna: Useful Features for Research

## Installation
```bash
# PyPI
pip install optuna
# Anaconda Cloud
conda install -c conda-forge optuna
# Install Optuna Dashboard
pip install optuna-dashboard
```

## Study Management & Storage
- **Create and persist studies with SQLite:**
```python
study = optuna.create_study(study_name="foo_study", storage="sqlite:///example.db")
study.optimize(objective)
```
- **Save and resume studies with joblib:**
```python
import joblib
study = optuna.create_study()
joblib.dump(study, "study.pkl")
# Later
study = joblib.load("study.pkl")
```

## Objective Functions
- **Custom arguments via callable class or lambda:**
```python
class Objective:
    def __init__(self, min_x, max_x):
        self.min_x = min_x
        self.max_x = max_x
    def __call__(self, trial):
        x = trial.suggest_float("x", self.min_x, self.max_x)
        return (x - 2) ** 2
study.optimize(Objective(-100, 100), n_trials=100)
```
- **Lambda wrapper:**
```python
def objective(trial, min_x, max_x):
    x = trial.suggest_float("x", min_x, max_x)
    return (x - 2) ** 2
study.optimize(lambda trial: objective(trial, -100, 100), n_trials=100)
```

## Reproducibility
- **Set random seed for deterministic results:**
```python
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
```

## Storage & Parallelization
- **Reliable parallelization with JournalFileBackend:**
```python
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
storage = JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
study = optuna.create_study(storage=storage)
```
- **Heartbeat monitoring for trial failure detection:**
```python
storage = optuna.storages.RDBStorage(url="sqlite:///:memory:", heartbeat_interval=60, grace_period=120)
study = optuna.create_study(storage=storage)
```
- **Retry failed trials automatically:**
```python
from optuna.storages import RetryFailedTrialCallback
storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory:",
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
)
study = optuna.create_study(storage=storage)
```

## Artifact Management
- **Save and retrieve models with ArtifactStore:**
```python
import os, pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)
def objective(trial):
    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    clf = SVC(C=svc_c)
    clf.fit(X_train, y_train)
    with open("model.pickle", "wb") as fout:
        pickle.dump(clf, fout)
    artifact_id = optuna.artifacts.upload_artifact(
        artifact_store=artifact_store,
        file_path="model.pickle",
        study_or_trial=trial.study,
    )
    trial.set_user_attr("artifact_id", artifact_id)
    return 1.0 - accuracy_score(y_valid, clf.predict(X_valid))
# List all models
for artifact_meta in optuna.artifacts.get_all_artifact_meta(study_or_trial=study):
    print(artifact_meta)
# Download the best model
trial = study.best_trial
best_artifact_id = trial.user_attrs["artifact_id"]
optuna.artifacts.download_artifact(
    artifact_store=artifact_store,
    file_path='best_model.pickle',
    artifact_id=best_artifact_id,
)
```

## Logging & Callbacks
- **Suppress log messages:**
```python
optuna.logging.set_verbosity(optuna.logging.WARNING)
```
- **Custom logging callback for best value updates:**
```python
def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(f"Trial {frozen_trial.number} finished with best value: {frozen_trial.value} and parameters: {frozen_trial.params}")
study.optimize(objective, n_trials=100, callbacks=[logging_callback])
```

## Dashboard & Visualization
- **Launch dashboard:**
```bash
optuna-dashboard sqlite:///db.sqlite3
```
- **Matplotlib/Plotly visualizations:**
```python
import optuna.visualization as vis
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
```

## Integration
- **Integration modules:**
  - TensorBoard, TensorFlow, Keras, Weights & Biases, XGBoost, etc.
  - See: [Optuna Integrations](https://github.com/optuna/optuna/tree/master/optuna/integration)

## Advanced Tips
- **Avoid duplicate parameter evaluation:**
```python
from optuna.trial import TrialState
def objective(trial):
    x = trial.suggest_int("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    return x ** 2 + y ** 2
```
- **Optimize over permutations (Lehmer code):**
```python
lehmer_code = [trial.suggest_int(f"x{i}", 0, n - i - 1) for i in range(n)]
# decode Lehmer code to permutation
```
- **Garbage collection after trial to avoid OOM:**
```python
study.optimize(objective, n_trials=10, gc_after_trial=True)
# or
import gc
study.optimize(objective, n_trials=10, callbacks=[lambda study, trial: gc.collect()])
```

---

For more, see the [Optuna documentation](https://optuna.readthedocs.io/) and [GitHub repo](https://github.com/optuna/optuna). 