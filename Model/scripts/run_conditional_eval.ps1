param(
  [string]$OutDir = "Model/experiments/outputs/cond_eval",
  [int]$NumSamples = 4,
  [int]$BatchSize = 1,
  [int]$NumSteps = 50,
  [int]$HiddenDim = 128,
  [int]$NumLayers = 4,
  [string]$Device = "cpu"
)

$DatasetPath = "Model/src/data/crossdocked_v1.1_rmsd1.0_processed"
$SplitFile = "Model/src/data/split_by_name.pt"

python Model/main.py --mode sample_conditional --dataset_path $DatasetPath --split_file $SplitFile --out_dir $OutDir --batch_size $BatchSize --num_workers 0 --num_samples $NumSamples --num_layers $NumLayers --hidden_dim $HiddenDim --num_steps $NumSteps --device $Device


