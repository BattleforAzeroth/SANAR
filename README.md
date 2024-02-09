## Non-Autoregressive Line-Level Code Completion


### Requirements
* Python >= 3.7
* Pytorch >= 1.5.0
* Fairseq 1.0.0a0


### Dataset

Path: /data_processing/data_source/

PY150: The python corpus data is collected by [Probabilistic Model for Code with Decision Trees](https://dl.acm.org/doi/pdf/10.1145/2983990.2984041), ant it contains 100,000 training programs and 50,000 testing programs.

JAVA-github: The java corpus data is collected by [Mining Source Code Repositories at Massive Scale using Language Modeling](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6624029). We follow [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](https://arxiv.org/pdf/2003.07914.pdf) to split the valid and test projects, and sample other 1k projects for training projects.


### Preparation

- Run `data_processing/python_tokenizing.py` or `data_processing/java_tokenizing.py` to tokenize the programs and save the token and type data instance.


Binarize the training data.

```
input_dir=path_to_raw_text_data
save_dir=path_to_binarized_output
src=source_languages
tgt=target_language

token-type:

python3 fairseq_cli/preprocess.py --source-lang xtoken --target-lang ytype --nwordstgt 50000 --nwordssrc 50000  --trainpref ${input_dir}/train --testpref ${input_dir}/eval --destdir ${save_dir} --workers 60 

token-token:

python3 fairseq_cli/preprocess.py --source-lang xtoken --target-lang ytoken --nwordstgt 50000 --nwordssrc 50000  --trainpref ${input_dir}/train --testpref ${input_dir}/test --validpref ${input_dir}/eval --destdir ${save_dir} --workers 60 --joined-dictionary
```

### Train


data_path=path_to_your_data

checkpoint_path=path_to_your_checkpoint

```
cd SANAR/

python3 fairseq_cli/train.py ${path_to_your_data} --user-dir sanar_plugins --noise full_mask --share-all-embeddings \
--source-lang xtoken --target-lang ytoken \
--label-smoothing 0.1 --lr 5e-5 \
--warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
--lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam \
--adam-betas '(0.9, 0.999)' \
--adam-eps 1e-6 --task translation_lev_modified --max-tokens 16384 \
--weight-decay 0.01 --dropout 0.1 \
--encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 \
--decoder-embed-dim 512 --fp16 \
--max-source-positions 1000 --max-target-positions 1000 \
--max-epoch 30 --seed 0 --clip-norm 5 \
--save-dir ${checkpoint_path} \
--src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \         
--eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
--eval-bleu-remove-bpe --best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
--arch glat --criterion glat_loss 
--activation-fn gelu \
--valid-subset valid \
--skip-invalid-size-inputs-valid-test --max-tokens-valid 4096 \
```

### Inference

```
python3 fairseq_cli/generate.py ${path_to_your_data} \
--path ${checkpoint_path}checkpoint_best.pt --user-dir sanar_plugins \
--task translation_lev_modified --remove-bpe --max-tokens 2048 \
--source-lang xtoken --target-lang ytoken \
--iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
--iter-decode-with-beam 1 --gen-subset test \
--skip-invalid-size-inputs-valid-test --quiet

```

Our implementation is based on [GLAT](https://arxiv.org/pdf/2008.07905.pdf).
