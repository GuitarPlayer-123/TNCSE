# TNCSE
Tensor’s Norm Constraints for Unsupervised Contrastive Learning of Sentence Embeddings

# Code and Saved Checkpoints of TNCSE

## Dear reviewers, codes and checkpoints for TNCSE training and validation are provided in this repository; please check them.

### Environmental Requirements
```
Pytorch=1.7.1 Transformers=2.4.1
```
## Direct Evaluation

#### We report the results directly below the command.

### Eval TNCSE_BERT

>python evaluation_CKPT.py --model_name_or_path_1 TNCSE_BERT_CKPT/BERT_1 --model_name_or_path_2 TNCSE_BERT_CKPT/BERT_2


    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 75.52 | 83.91 | 77.57 | 84.97 | 80.42 |    81.72     |      72.97      | 79.58 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_RoBERTa

>python evaluation_CKPT.py --model_name_or_path_1 TNCSE_RoBERTa_CKPT/RoBERTa1 --model_name_or_path_2 TNCSE_RoBERTa_CKPT/RoBERTa1

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 74.11 | 84.01 | 76.07 | 84.80 | 81.60 |    82.68     |      73.47      | 79.53 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_BERT_D

>python evaluation_D.py --model_name_or_path TNCSE_D_BERT

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 75.42 | 84.64 | 77.62 | 84.92 | 80.50 |    81.79     |      73.52      | 79.77 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_RoBERTa_D

>python evaluation_D.py --model_name_or_path TNCSE_D_RoBERTa

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 74.56 | 84.74 | 76.30 | 84.89 | 81.70 |    83.01     |      74.18      | 79.91 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_BERT_UC

>python ensemble_UC.py --model_type BERT

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 75.80 | 85.27 | 78.67 | 85.99 | 82.01 |    83.16     |      73.01      | 80.56 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_RoBERTa_UC

>python ensemble_UC.py --model_type RoBERTa

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 74.52 | 85.26 | 77.63 | 85.85 | 82.62 |    83.65     |      73.35      | 80.41 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


### Eval TNCSE_BERT_UC_D

>python evaluation_D.py --model_name_or_path TNCSE_UC_D_BERT

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 75.94 | 85.31 | 78.50 | 85.69 | 81.86 |    83.03     |      73.89      | 80.60 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+



### Eval TNCSE_RoBERTa_UC_D
>python evaluation_D.py --model_name_or_path TNCSE_UC_D_RoBERTa

    ------ test ------
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+
    | 74.14 | 83.86 | 76.08 | 84.06 | 81.59 |    82.90     |      73.55      | 79.45 |
    +-------+-------+-------+-------+-------+--------------+-----------------+-------+


## Train TNCSE

### Data Preparation

We have prepared the unlabelled training set, located in **data/Wiki_for_TNCSE.txt**; the seven STS test sets are contained in **SentEval**.

### Pre-training Model

The checkpoints we need to use for training TNCSE-BERT and TNCSE-RoBERTa are saved in **TNCSE_BERT_encoder1**, **TNCSE_BERT_encoder2**, **TNCSE_RoBERTa_encoder1**, and **TNCSE_RoBERTa_encoder2**, respectively, which are RTT Data Augmentation and unsupervised SimCSE trained.

### Train TNCSE_BERT

>python train_dual.py

### Train TNCSE_RoBERTa

>python train_dual.py --output_path TNCSE_RoBERTa_OUTPUT --pretrain_model_path_1 TNCSE_RoBERTa_encoder1 --pretrain_model_path_2 TNCSE_RoBERTa_encoder2 --pretrain_tokenizer Roberta-base --batch_size_train 256 --dropout 0.0175 --lr 1e-06
