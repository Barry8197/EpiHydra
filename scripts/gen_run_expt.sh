for lr_lora in 0.0001 0.001 0.01; do
    for lora_r in 8 16 32; do
        lora_alpha=$((lora_r * 2))
        echo "python -m epihydra.run --data_path /work/gr-fe/bryan/EpiHydra/data/03_final/combined/methylation.pkl --model_dir /work/gr-fe/bryan/EpiHydra/data/03_final/methylgpt/medium --log_dir /work/gr-fe/bryan/EpiHydra/results/tensorboard/runs --save_dir /work/gr-fe/bryan/EpiHydra/results/tensorboard/checkpoints --epochs 100 --batch_size 32 --lr_lora $lr_lora --lora_r $lora_r --lora_alpha $lora_alpha --mask_ratio 0.3 --dropout 0.3"
    done
done > experiments.txt
