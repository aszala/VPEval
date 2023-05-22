mkdir -p results

models=("sd_14" "sd_21" "karlo" "mindalle" "dallemega" "vpgen_gligen")
for model in ${models[*]}; do
    python src/skill_based_main.py \
        --image_dir ./data/images/skill_based/ \
        --metadata_path ./data/skill_based/ \
        --model ${model} \
        ${@:1}
done