python train.py \
  --data-dir data/100 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 500000 --optimizer adam --lr 1e-3 \
  --phy-batch-size 31000 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cuda --dtype float32 --seed 1234 \
  --log-every 50 --save-every 1000 --eval-batch-size 1024 \
  --output-root runs --run-tag linux-baseline-100