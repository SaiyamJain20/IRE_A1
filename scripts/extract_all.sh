#!/bin/bash

# Extract EBNerd
unzip -o data/ebnerd_demo.zip -d data/ebnerd_demo
unzip -o data/ebnerd_small.zip -d data/ebnerd_small

# Extract MIND
unzip -o data/MINDsmall_train.zip -d data/MINDsmall_train
unzip -o data/MINDsmall_dev.zip -d data/MINDsmall_dev

# Extract Embeddings
unzip -o data/Ekstra_Bladet_word2vec.zip -d data/embeddings/ebnerd_word2vec
unzip -o data/google_bert_base_multilingual_cased.zip -d data/embeddings/bert_multilingual

echo "Extraction complete!"
