mkdir -p results

models=("sd_14" "sd_21" "karlo" "mindalle" "dallemega" "vpgen_gligen")
for model in ${models[*]}; do
    datasets=("paintskill" "drawbench" "parti" "coco")
    for dataset in ${datasets[*]}; do
        python src/open_ended_main.py \
            --image_dir ./data/images/open_ended/ \
            --prompt_file ./data/open_ended/model_dataset_specific_prompts/${model}_${dataset}.json \
            --savefile ./results/${model}_${dataset}.json \
            ${@:1}
    done
done