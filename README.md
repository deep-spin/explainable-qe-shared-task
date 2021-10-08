# Minimalist explainable XLM-R QE system 

This repo contains the code for the [IST-Unbabel 2021 Submission for the Quality Estimation Shared Task](https://eval4nlp.github.io/sharedtask.html).  


### Data from the shared task:

Preprocess the entire MLQE-PE dataset into the shared task format and download/preprocess the test set:
```bash
mkdir data
git clone https://github.com/sheffieldnlp/mlqe-pe
bash preprocess_mlqepe.sh
python3 preprocess_mlqepe.py --input-dir mlqe-pe/data/ --output-dir data/
bash download_and_preprocess_test_data.sh
rm -rf mlqe-pe
```


### Installation:

```bash
pip install -r requirements.txt
pip install -e .
```


### Training:

Inform a config file via `-f`:

```bash
python3 cli.py train -f configs/xlmr-adapters-shared-task-mlqepe-all-all.yaml
```

See more config files in the `config/` folder. PyTorch models can be found in the `model/` folder. 


### Evaluating:

```bash
python3 scripts/evaluate_sentence_level.py train --testset data/ro-en/dev --checkpoint path/to/model.ckpt
```

For word-level models:
```bash
python3 scripts/evaluate_word_level.py train --testset data/ro-en/dev --checkpoint path/to/model.ckpt
```


### Extracting explanations:

For baseline explainers (gradient, leave-one-out, etc.), use `explain.py`. For example:
```bash
python3 scripts/explain.py
  --testset data/ro-en/dev
  --checkpoint path/to/model.ckpt
  --explainer ig
  --save experiments/explanations/roen_ig/
  --batch-size 1
```

For extracting attention, use `explain_attn.py`. For example:
```bash
python3 scripts/explain_attn.py
  --testset data/ro-en/dev
  --checkpoint path/to/model.ckpt
  --save experiments/explanations/roen_attn
  --batch-size 1
```

Several folders will be created with their name prefixed by the path informed via the flag `--save` for: 
- the entire model (average layers via scalar mix)
- for each layer (average heads)
- for each head (average the "rows" in the attention map)

Moreover, if you want to get explanations in terms of `attention * norm(values)`, inform can inform these flags:
```bash
  --norm-attention
  --norm-strategy weighted_norm
```


We also provide scripts for extracting explanations with other methods, e.g., [DiffMask](https://github.com/nicola-decao/diffmask) and [Attention Flow/Rollout](https://github.com/samiraabnar/attention_flow).

### Evaluating explanations

Use the script `evaluate_explanations.py`. For example:
```bash
python3 scripts/evaluate_explanations.py
  --gold_sentence_scores_fname data/et-en/dev.da
  --gold_explanations_fname_mt data/et-en/dev.tgt-tags
  --gold_explanations_fname_src data/et-en/dev.src-tags
  --model_sentence_scores_fname experiments/explanations/eten_attn_head_18_3/sentence_scores.txt
  --model_explanations_fname_mt experiments/explanations/eten_attn_head_18_3/mt_scores.txt
  --model_explanations_fname_src experiments/explanations/eten_attn_head_18_3/source_scores.txt
  --model_fp_mask_mt experiments/explanations/eten_attn_head_18_3/mt_fp_mask.txt
  --model_fp_mask_src experiments/explanations/eten_attn_head_18_3/source_fp_mask.txt
  --reduction sum
  --transform none
```

The `--reduction` flag informs how to aggregate word pieces scores: `none, first, sum, mean, max`. 
The flag `--transform` can be:
- `pre`: apply `sigmoid(abs(.))` element-wise for each score BEFORE aggregating word pieces
- `pos`: apply `sigmoid(abs(.))` element-wise for each score AFTER aggregating word pieces
- `none`: do not apply any transformation

This transformation might be useful for explainers that can return negative values. 
The computation of sigmoid is in fact irrelevant, since the metrics are based on ranking. But it is useful to have 
scores between 0 and 1 if we want to do some kind of thresholding to calculate accuracy or something else.


### Prepare Submission:
Aggreagte subword units:
```bash
python3 scripts/aggregate_explanations.py \
  --model_explanations_dname experiments/explanations/roen_ig/ \
  --reduction sum \
  --transform none
```

Create a metadata.txt file, and zip all files. Here is the script that does all of this:
```bash
python3 scripts/prepare_submission.py
  --explainer experiments/explanations/roen_ig/
  --save submission.zip
  --team "Team Name"
  --track "constrained"
  --desc "Simple description of the model + explainer."
```

A file called `submissions.zip` will be created in the working directory with the explanations of `ig` for `ro-en`.


### Bibtex entry
```
@inproceedings{treviso-et-al-2021-ist,
  title        = {IST-Unbabel 2021 Submission for the Explainable Quality Estimation Shared Task},
  author       = {Treviso, Marcos and Guerreiro, Nuno and Rei, Ricardo and Martins, Andr{\'e} F. T.},
  year         = 2021,
  month        = nov,
  booktitle    = {Proceedings of the Second Eval4NLP Workshop on Evaluation and Comparison of NLP Systems},
  publisher    = {Association for Computational Linguistics},
  address      = {Online},
}
```

### License

MIT.

