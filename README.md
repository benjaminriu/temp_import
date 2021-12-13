# MLRNetTabularBenchmark
### Clone repository
```bash
export REPO_DIR=<ABSOLUTE path to the desired repository directory>
git clone <repository url> $REPO_DIR
cd $REPO_DIR
```

- Setup.ipynb to set up repositories and env:

### Set up directories
```bash
mkdir ../preprocessed_datasets
mkdir outputs
mkdir predictions
mkdir ../feature_reorder_index
mkdir ../reordered_datasets
```
## Quick Demo
```bash
conda create -n mlrnet_demo python=3.7
conda activate mlrnet_demo
conda install ipykernel pandas=1.1.3 scikit-learn=0.23.2 -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda deactivate
```
- Demo_MLRNet.ipynb to get a quick demo on boston


## Benchmark
```bash
conda create -n mlrnet_benchmark python=3.7
conda activate mlrnet_benchmark
conda install ipykernel pandas=1.1.3 scikit-learn=0.23.2 -y
conda install -c conda-forge sklearn-contrib-py-earth=0.1.0 xgboost=1.3.3 catboost=0.26.1 lightgbm=3.2.1 optuna=2.10.0 -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda deactivate
```

```bash
python download_data.py '../preprocessed_datasets/'
python run_benchmark.py
```
- Run_Benchmark.ipynb to run a set of experiments
