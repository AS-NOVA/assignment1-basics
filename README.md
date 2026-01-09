# CS336 Spring 2025 Assignment 1: Basics

### 运行方法

快速测试
```sh
python ./scripts/my_train.py \
    --total_iters 100 \
    --eval_interval 10 \
    --eval_iters 10 \
    --batch_size 16 \
    --not_save_model \
    --wandb_project TinyStories_17M_test \
    --wandb_run_name fasttest
```

正式使用，学习率调试（无需保存模型）
```sh
python ./scripts/my_train.py \
    --lr_max 0.01 \
    --not_save_model \
    --wandb_project TinyStories_17M_lr_tuning
```



### Run unit tests


```sh
uv run pytest
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

