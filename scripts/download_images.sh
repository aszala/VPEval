mkdir -p data/

wget https://huggingface.co/datasets/abhayzala/VPEval/resolve/main/generated_images.tar.gz -O ./data/generated_images.tar.gz

tar -xzvf ./data/generated_images.tar.gz -C ./data/

rm -rf ./data/generated_images.tar.gz