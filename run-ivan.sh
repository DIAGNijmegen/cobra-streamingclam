cd /data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam
python3 fetch_data.py
FOLD="0" 
python3 main.py \
    --image_path=/data_source/images \
    --mask_path=../data_source/images_tissue_masks \
    --fold="${FOLD}" \
    --train_csv="../data_source/train.csv" \
    --val_csv="../data_source/val.csv" \
    --test_csv="../data_source/test.csv" \
    --mask_suffix="_tissue" \
    --mode="fit" \
    --unfreeze_streaming_layers_at_epoch=20 \
    --num_epochs=40 \
    --strategy="ddp_find_unused_parameters_true" \
    --default_save_dir="/data/pathology/projects/DeepDerma/BCC_SCLAM/debug/ckpt" \
    --ckp_path="" \
    --grad_batches=1 \
    --num_gpus=1 \
    --precision="bf16-mixed" \
    --encoder="resnet39" \
    --branch="sb" \
    --pooling_layer="avgpool" \
    --pooling_kernel=16 \
    --num_classes=2 \
    --learning_rate=1e-4 \
    --tile_size=9600 \
    --tile_size_finetune=6400 \
    --image_size=65536 \
    --filetype=".tif" \
    --read_level=1 \
    --num_workers=3
