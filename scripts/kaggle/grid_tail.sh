#!/bin/bash
export MPLBACKEND=Agg
mkdir -p output output/nb2_jpeg
python run_lora_simple_colab.py --corruption jpeg --layers drift --seed 42 --drift-csv output/nb2_jpeg/cka_matrix.csv --outdir output/jpeg_drift > output/jpeg_drift.log 2>&1
echo "drift jpeg exit $?"
echo "===== CHAIN contrast START ====="
python run_notebook2.py --corruption contrast --output output/nb2_contrast > output/nb2_contrast.log 2>&1
echo "nb2 contrast exit $?"
python run_lora_simple_colab.py --corruption contrast --layers all --seed 42 --outdir output/contrast_uniform > output/contrast_uniform.log 2>&1
echo "uniform contrast exit $?"
n=$(grep -c ",5," output/nb2_contrast/cka_matrix.csv 2>/dev/null || echo 0)
if [ "$n" -eq 12 ]; then
  python run_lora_simple_colab.py --corruption contrast --layers drift --seed 42 --drift-csv output/nb2_contrast/cka_matrix.csv --outdir output/contrast_drift > output/contrast_drift.log 2>&1
  echo "drift contrast exit $?"
else
  echo "DRIFT contrast SKIPPED ($n/12)"
fi
