# float64 run (reference)
# pixi run python scripts/optimize_density.py -i input_data.h5 -o result.h5 \
#     -n 250 \
#     --lr_box 1.0 --lr_phi 0.1 \
#     --n_inner_phi 2000 --n_inner_box 10 \
#     --tol_grad_phi 1e-6 --tol_grad_box 1e-6 \
#     --log_every 50 --box_grad_scale 10 \
#     > optimizer.log 2>&1

# float32 run (precision test)
pixi run python scripts/optimize_density.py -i input_bcc.h5 -o ./result_bcc.h5 \
    --dtype float32 \
    -n 50 \
    --lr_box 1.0 --lr_phi 1.0 \
    --n_inner_phi 250 --n_inner_box 20 \
    --log_every 10 --box_grad_scale 10 \
     # > optimizer_float32.log 2>&1
