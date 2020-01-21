# Single Path One-Shot NAS
Here is a pytorch based re-implementation of the paper: [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/pdf/1904.00420.pdf), arXiv 2019, referring to [megvii-model/ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series).

The choice block structure, data augmentation strategy, hyperparameters, etc., are based on ShuffleNetV2+. We only implement the block search part.

## Environments
```shell
python == 3.7.3 pytorch == 1.3.1 cuda == 10.1
pip install thop

git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install --cpp_ext
```

## Usage 
### Supernet Training and Architecture Search
```shell
sh distributed_arch_search.sh NUM_GPU or python arch_search.py \
  --gpu_devices AVAILABLE_GPU_DEVICES \
  --total_epochs TOTAL_EPOCHS \
  --batch_size BATCH_SIZE_PER_GPU (distributed) or TOTAL_BATCH_SIZE (non-distributed) \
  --lr LR \
  --pop_size POPULATION_SIZE \
  --total_search_iters TOTAL_SEARCH_ITERS \
  --mut_prob MUTATION_PROB \
  --topk TOPK_POPULATION \
  --resume_path /PATH/TO/RESUME/CHECKPOINT \
  --history_path /PATH/TO/SEARCH/HISTORY \
  --raw_train_dir /DIR/TO/IMAGENET/TRAIN/DATASET
```

### Best Architecture Training
```shell
sh distributed_best_arch_train.sh NUM_GPU or python best_arch_train.py \
  --gpu_devices AVAILABLE_GPU_DEVICES \
  --best_arch BEST_ARCHITECTURE \
  --total_epochs TOTAL_EPOCHS \
  --batch_size BATCH_SIZE_PER_GPU (distributed) or TOTAL_BATCH_SIZE (non-distributed) \
  --lr=LR \
  --resume_path /PATH/TO/RESUME/CHECKPOINT \
  --train_dir /PATH/TO/IMAGENET/TRAIN/DATASET \
  --val_dir /PATH/TO/IMAGENET/VAL/DATASET
```

## Results
### Hyperparameters
#### supernet training
lr = 0.5, total_epochs = 120, batch_size = 400, weight_decay = 4e-5, label_smooth = 0.1
#### architecture search
population_size = 50, total_search_iters = 20, mutation_prob = 0.1, topk_population = 10
#### best architecture training
lr = 0.5, total_epochs = 240, batch_size = 400, weight_decay = 4e-5, label_smooth = 0.1

### Architecture Search Results
Top-10 architectures:

| Arch | FLOPs | Prec@1 (%) | Prec@5 (%) |
| --- | :---: | :---: | :---: |
| [1, 0, 3, 3, 2, 2, 1, 0, 1, 3, 2, 1, 3, 3, 1, 0, 3, 0, 0, 1] | 328M | 62.48 | 84.29 |
| [1, 2, 3, 3, 2, 1, 1, 0, 1, 2, 2, 2, 3, 3, 1, 2, 3, 0, 0, 1] | 328M | 62.47 | 84.38 |
| [1, 0, 3, 3, 2, 2, 1, 0, 1, 3, 2, 2, 3, 3, 1, 0, 3, 0, 0, 1] | 329M | 62.47 | 84.28 |
| [1, 0, 3, 3, 2, 1, 1, 0, 1, 2, 1, 2, 3, 3, 1, 2, 3, 0, 0, 1] | 323M | 62.46 | 84.45 |
| [1, 0, 3, 3, 2, 1, 1, 0, 1, 3, 2, 1, 3, 3, 1, 0, 3, 0, 0, 1] | 326M | 62.46 | 84.30 |
| [1, 0, 3, 3, 2, 1, 1, 0, 1, 3, 1, 2, 3, 3, 1, 2, 3, 0, 0, 1] | 328M | 62.46 | 84.38 |
| [1, 3, 3, 3, 2, 1, 1, 0, 1, 2, 1, 2, 3, 3, 1, 2, 3, 0, 0, 1] | 328M | 62.45 | 84.44 |
| [1, 0, 3, 3, 2, 2, 1, 0, 1, 3, 1, 1, 3, 3, 0, 0, 3, 0, 0, 1] | 327M | 62.45 | 84.30 |
| [1, 2, 3, 3, 2, 1, 1, 0, 1, 2, 1, 2, 3, 3, 1, 2, 3, 0, 0, 1] | 327M | 62.44 | 84.38 |
| [1, 0, 3, 3, 2, 1, 0, 0, 2, 2, 1, 2, 3, 3, 1, 3, 3, 0, 0, 1] | 328M | 62.44 | 84.44 |

### Best Architecture Training Results
| Model | FLOPs |  Prec@1 (%) | Prec@5 (%) |
| --- | :---: | :---: | :---: |
| Best architecture | 328M | ? | ? |
| Official | 319M | 74.3 | - |
