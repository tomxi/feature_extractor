# feature_extractor
script for extracting multiple audio representations.

## Usage:
Setup environment with conda:

```
conda env create -f extract_env.yml
conda activate extract
```

Run the script on all audio files in a directory
```
python extract_features.py /path/to/input/audio/directory /path/to/feature/output/directory [--recompute]
```

Use the optional flag `--recompute` to force feature computation on all files in the audio directory. 
Otherwise, skip existing files in the feature output directory.
