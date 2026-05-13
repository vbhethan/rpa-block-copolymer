# float32 run
pixi run python scripts/optimize_density.py -i input.h5 -o ./result.h5\
    --dtype float32 \
    -n 50 \
    --lr_box 1.0 --lr_phi 1.0 \
    --n_inner_phi 2000 --n_inner_box 20 \
    --log_every 10 --box_grad_scale 10 \
    --tol_grad_phi 1e-4 --tol_grad_box 1e-5
