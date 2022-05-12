# Spring 2022 CSE 354 - Natural Language Processing Final Project -- Sentence Representation Classification
Tareq Mia, Jackie Li, Dennis Chan

## Original Source
This project is an extension of the Natural Language Processing Assignment 2. Therefore, the original code base of that assignment was used for this project.

## Files Modified
The following files from the original codebase were modified for this project: 
1. `sequence_to_vector.py`
2. `main_model.py`
3. `train.py`
4. `plot_performance_against_data_size.py`
5. `plot_perturbation_analysis.py`
6. `plot_probing_performances_on_bigram_order_task.py`
7. `plot_probing_performances_on_sentiment_task.py`

## Commands 
### Retrieving the Dataset
For the GloVE wordvectors,
`./download_glove.sh`


### CNN Model
For training the CNN model for IMDB Sentiment 5k, 10 and 15k, run the following commands:  
* `python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice cnn --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _cnn_5k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`
* `python train.py main data/imdb_sentiment_train_10k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice cnn --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _cnn_10k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`
* `python train.py main data/imdb_sentiment_train_15k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice cnn --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _cnn_15k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`

### DAN Model
For training the DAN model with attention weights for IMDB Sentiment 5k, 10 and 15k, , run the following commands:  
* `python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 5 --suffix-name _dan_5k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`

* `python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 50 --suffix-name _dan_5k_with_emb_for_50k --pretrained-embedding-file data/glove.6B.50d.txt`

* `python train.py main data/imdb_sentiment_train_10k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 8 --suffix-name _dan_10k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`

* `python train.py main data/imdb_sentiment_train_15k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 8 --suffix-name _dan_15k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt` 

### BiLSTM Model
For training the BiLSTM for IMDB Sentiment 5k, 10k and 15k, run the following commands:
* `python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice bilstm --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _bilstm_5k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`
* `python train.py main data/imdb_sentiment_train_10k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice bilstm --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _bilstm_10k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`
* `python train.py main data/imdb_sentiment_train_15k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice bilstm --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _bilstm_15k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt`

### Probing

For training the probing models for the CNN model,
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_cnn_5k_with_emb --layer-num 1 --num-epochs 4 --suffix-name _sentiment_cnn_with_emb_on_5k_at_layer_1`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_cnn_5k_with_emb --layer-num 2 --num-epochs 4 --suffix-name _sentiment_cnn_with_emb_on_5k_at_layer_2`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_cnn_5k_with_emb --layer-num 3 --num-epochs 4 --suffix-name _sentiment_cnn_with_emb_on_5k_at_layer_3`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_cnn_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _sentiment_cnn_with_emb_on_5k_at_layer_4`
* `serialization_dirs/main_cnn_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _bigram_order_cnn_with_emb_on_5k_at_layer_4`

For training the probing models for the DAN model,
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 1 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_1`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 2 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_2`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 3 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_3`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 4 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_4`
* `python train.py probing data/bigram_order_train.jsonl data/bigram_order_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 4 --num-epochs 8 --suffix-name _bigram_order_dan_with_emb_on_5k_at_layer_4`

For training the probing models for the BiLSTM model,
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_bilstm_5k_with_emb --layer-num 1 --num-epochs 4 --suffix-name _sentiment_bilstm_with_emb_on_5k_at_layer_1`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_bilstm_5k_with_emb --layer-num 2 --num-epochs 4 --suffix-name _sentiment_bilstm_with_emb_on_5k_at_layer_2`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_bilstm_5k_with_emb --layer-num 3 --num-epochs 4 --suffix-name _sentiment_bilstm_with_emb_on_5k_at_layer_3`
* `python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_bilstm_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _sentiment_bilstm_with_emb_on_5k_at_layer_4`
* `python train.py probing data/bigram_order_train.jsonl data/bigram_order_dev.jsonl --base-model-dir serialization_dirs/main_bilstm_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name_bigram_order_bilstm_with_emb_on_5k_at_layer_4`


### Predicting

For predictions with the CNN based probing model,
* `python predict.py serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_1 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_1/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_2 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_2/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_3 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_3/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_4 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_cnn_with_emb_on_5k_at_layer_4/predictions_imdb_sentiment_5k_test.txt`

For predictions with the DAN based probing model,
* `python predict.py serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_1 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_1/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_2 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_2/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_3 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_3/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_4 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_dan_with_emb_on_5k_at_layer_4/predictions_imdb_sentiment_5k_test.txt`

For predictions with the BiLSTM based probing model,
* `python predict.py serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_1 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_1/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_2 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_2/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_3 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_3/predictions_imdb_sentiment_5k_test.txt`
* `python predict.py serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_4 data/imdb_sentiment_test.jsonl --predictions-file serialization_dirs/probing_sentiment_bilstm_with_emb_on_5k_at_layer_4/predictions_imdb_sentiment_5k_test.txt`


### Plots
There are four scripts in the code that were used to plot and analyze the sentence representations:
1. `plot_performance_against_data_size.py`
2. `plot_probing_performances_on_sentiment_task.py`
3. `plot_probing_performances_on_bigram_order_task.py`
4. `plot_perturbation_analysis.py`

The scripts will tell you the models that are missing and the corresponding commands needed before the analysis plot can be created.  
