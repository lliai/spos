# Single Path One-Shot NAS
Here is a pytorch based re-implementation of the paper: [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/pdf/1904.00420.pdf), arXiv 2019, referring to [megvii-model/ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series).

The data augmentation strategy, hyperparameters, etc., are the same with ShuffleNetV2+. We only implement the block search part.

## Environments
```shell
python == 3.7.3 pytorch == 1.3.1 cuda == 10.1
pip install thop
```

## Usage 
### Dataset Preparation
First collect the train dataset and val dataset for supernet training and architecture search. We split the original imagenet training set into two parts: 50000 images for validation (50 images for each class exactly) and the rest as the training set. You can run the following script: 
```shell
python data_split.py
```
Then all the absolute paths of the splitted train dataset and val dataset will be saved into csv files.

### Supernet Training and Architecture Search
```shell
sh distributed_arch_search.sh NUM_GPU or python arch_search.py
```

### Best Architecture Training
```shell
sh distributed_best_arch_train.sh NUM_GPU or python best_arch_train.py
```

## Results
### Architecture Search Results
Top-10 architectures:

| Arch | FLOPs | Prec@1 | Prec@5 |
| --- | :---: | :---: | :---: |
| [2, 2, 3, 0, 2, 0, 2, 1, 2, 2, 2, 1, 3, 1, 1, 3, 2, 0, 1, 1] | 339M | 62.82 |
| [2, 3, 1, 0, 2, 2, 3, 1, 2, 2, 2, 1, 3, 3, 1, 3, 2, 0, 1, 1] | 348M | 62.78 |
| [2, 2, 3, 0, 2, 0, 1, 1, 2, 2, 2, 1, 3, 3, 1, 0, 2, 0, 1, 1] | 337M | 62.74 |
| [2, 2, 3, 0, 2, 1, 1, 1, 2, 2, 2, 1, 3, 1, 1, 3, 2, 0, 1, 1] | 339M | 62.74 |
| [2, 2, 3, 0, 2, 0, 1, 1, 2, 2, 2, 1, 3, 1, 1, 3, 2, 0, 1, 1] | 338M | 62.72 |
| [2, 2, 3, 0, 2, 3, 1, 1, 2, 3, 2, 1, 3, 3, 1, 3, 2, 0, 1, 1] | 353M | 62.70 |
| [2, 2, 3, 0, 1, 0, 1, 2, 2, 2, 2, 1, 3, 1, 1, 0, 2, 0, 1, 1] | 331M | 62.69 |
| [0, 2, 1, 0, 2, 0, 1, 2, 2, 2, 2, 1, 2, 3, 1, 3, 2, 0, 1, 1] | 331M | 62.66 |
| [2, 2, 3, 0, 2, 0, 1, 1, 2, 2, 2, 1, 3, 1, 1, 0, 2, 0, 1, 1] | 332M | 62.65 |
| [2, 2, 3, 0, 2, 0, 1, 2, 2, 2, 2, 1, 3, 3, 1, 3, 2, 0, 1, 1] | 345M | 62.65 |

### Supernet and Best Architecture Training Results
| Model | Params | FLOPs |  Prec@1 | Prec@5 |
| --- | :---: | :---: | :---: | :---: |
| Supernet | 10.6M | - | 61.4 | 83.6 |
| Best architecture | 3.8M | 339M | 72.7 | 91.0
| Official | 3.3M | 319M | 74.3 | - | 

## Logs
### Supernet Training
![](imgs/supernet_loss.png)
### Best Subnet Training
![](imgs/best_subnet_loss.png)
