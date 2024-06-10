# IceSpike Implementation

This repository contains the implementation of IceSpike, based on the original SpikeBERT implementation.

## Prerequisites

First, clone the original SpikeBERT repository from the authors:

```bash
git clone https://github.com/Lvchangze/SpikeBERT.git
```
## Main Files and Modifications
### 1. predistill_spikformer.py

This is the first stage of the knowledge distillation process. In the original, the authors use the BERT model, while we use IceBERT with RoBERTa architecture. Therefore, all imports and uses related to this were changed. Minor changes were made in processing datasets, and some parameters were removed accordingly.

We ran it with the following configurations:

    * 2.2.2+cu118
    * Seed: 42
    * Batch Size: 16
    * Fine-tune Learning Rate: 5e-05
    * Max Sample Number: 80462898
    * Epochs: 1
    * Label Number: 2
    * Depths: 6
    * Max Length: 256
    * Dimension: 768
    * Representation Weight: 1.0
    * Tau: 10.0
    * Common Threshold: 1.0
    * Number of Steps: 16
    * Teacher Model Path: mideind/IceBERT-ic3
    * Ignored Layers: [0, 1]
    * All Samples: 1970350
    * Skip Probability: 40.836855381023675

### 1. new_distill_spikformer.py

This is the second and final stage of the knowledge distillation. The same BERT-RoBERTa changes were made, along with minor changes in processing datasets.

Run with the following configurations:

    * 2.2.2+cu118
    * Use GPU: True
    * GPU Count: 2
    * GPU Type: NVIDIA A100-PCIE-40GB
    * Seed: 42
    * Dataset Name: igc_full
    * Batch Size: 32
    * Fine-tune Learning Rate: 0.0005
    * Epochs: 30
    * Teacher Model Path: /users/home/elenao23/Models_2/results/elenaovv/model_igc_full/checkpoint-45468
    * Label Number: 4
    * Depths: 12
    * Max Length: 128
    * Dimension: 768
    * Cross-Entropy Weight: 0.1
    * Embedding Weight: 0.1
    * Logit Weight: 1.0
    * Representation Weight: 0.1
    * Number of Steps: 4
    * Tau: 2.0
    * Common Threshold: 1.0
    * Predistill Model Path: /users/home/elenao23/SpikeBert3/SpikeBERT/saved_models/predistill_spikformer_common_crawl/_lr5e-05_seed42_batch_size16_depths6_max_length256_tau10.0_common_thr1.0
    * Ignored Layers: 0
    * Metric: mcc
    * Load Predistill Model: True

## Important Notes

 1. Provide the teacher and student model paths carefully.
 2. Pay attention to loading and processing datasets (format, location, loading).
 3. Ensure that the configurations match your environment and dataset requirements.

For any issues or further assistance, please refer to the original [SpikeBERT repository](https://github.com/Lvchangze/SpikeBERT) and feel free to contact me.
