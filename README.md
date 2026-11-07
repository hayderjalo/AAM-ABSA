# AAM-ABSA

Training and evaluation code for Arabic text–image multilabel aspect–sentiment prediction.
This repository contains source code only. Input data and trained weights are not distributed.

## Architecture

Frozen AraBERT (`aubmindlab/bert-base-arabertv02`) provides token embeddings and contextual features.
Two bidirectional LSTMs and a contextual projection form the text representation.
Frozen ImageNet ResNet-18 supplies 49 spatial visual tokens. Text and visual features are projected
to 128 dimensions, refined by separate AMID modules, and combined using HCF.
HCF uses four-head self-attention, bidirectional cross-attention, and pooling of four sequences.
A sigmoid multilabel head predicts aspect–sentiment combinations.

Configurations: `AMID_HCF`, `no_AMID`, `no_HCF`, and `text_only`.
The text-only configuration retains text AMID. Removing HCF replaces both attention stages with
pooled concatenation and a linear projection.

## Installation

Use Python 3.10 or newer. Install a compatible PyTorch/torchvision pair for your hardware,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependency ranges are compatibility targets, not a recovered lockfile of the original runtime.
GPU execution is recommended. CPU execution is supported but slower. Frozen text and image
features are cached in host memory, so memory requirements grow with the input size.
Internet access is required on the first run to download pretrained encoder weights.

## Input contract

Provide three UTF-8 CSV files: `absa_train.csv`, `absa_validation.csv`, and `absa_test.csv`.
Each must contain:

| Column | Meaning |
| --- | --- |
| `sample_id` | Unique identifier across all splits |
| `text` | Nonempty Arabic text |
| `image_filename` | Relative image path inside the specified image directory |
| `aspect_labels_json` | JSON list of objects with `aspect` and `sentiment` keys |

Allowed sentiments are `negative`, `neutral`, and `positive`.
Labels are derived from training aspects crossed with the three sentiments; validation/test
labels outside this vocabulary are rejected. The classifier adapts to vocabulary size.
No specific private collection or label inventory is embedded in this repository.

## Run

```bash
python validate_data.py --data-dir /path/to/splits --image-dir /path/to/images
python train.py --data-dir /path/to/splits --image-dir /path/to/images --output-dir outputs
```

Defaults: seed 42, five epochs, batch size 32, AdamW learning rate 0.0003,
gradient clipping 1.0, text length 96, image size 224×224, dropout 0.2.
All four variants run by default; select individual variants using `--variants`.
Change `--seed`, `--epochs`, or `--device` explicitly as required.

At each epoch the validation set selects a global threshold from 0.10 to 0.80 in 0.05 increments.
The checkpoint with the highest validation Joint Macro-F1 is retained. Threshold ties select
the larger threshold; checkpoint ties retain the earlier checkpoint. Test labels are not used
for checkpoint or threshold selection.

## Kaggle

Upload the source files or clone this repository into `/kaggle/working`, enable a GPU and
internet access, and attach your own inputs. In a notebook cell, run:

```python
!python -m pip install -r /kaggle/working/AAM-ABSA/requirements.txt
!python /kaggle/working/AAM-ABSA/train.py --data-dir /kaggle/input/YOUR_INPUT/splits --image-dir /kaggle/input/YOUR_INPUT/images --output-dir /kaggle/working/aam_outputs
```

Replace `YOUR_INPUT` and the source directory with your actual paths.

## Outputs and metrics

Each run writes a best checkpoint, training history, test probabilities and predictions,
per-label precision/recall/F1/support, sample-level exact matches, and learning curves.
The output root contains an ablation summary, comparison plot, configuration, and split audit.
An archive of the outputs is created alongside the output directory.
Outputs can contain input identifiers and labels: keep them private when sharing source code.

Joint Macro-F1, Macro-Precision, and Macro-Recall average over the complete label vocabulary,
using zero for undefined values. Joint Micro-F1 pools label decisions. Joint Accuracy is exact
label-set match, not the average accuracy of individual binary decisions.

The split audit checks exact text and image-content overlap; it does not detect all semantic
or perceptual near-duplicates. Users supply and document their own annotation provenance,
optionally using `--label-provenance`.

## Recalculate saved predictions

```bash
python evaluate.py --predictions outputs/AMID_HCF/predictions.npz --output-dir outputs/recalculated
```

This also writes TP, FP, FN, and TN for each label.

## Attribution

Pretrained encoders are distributed by their respective authors. Follow the model providers'
licenses and cite AraBERT and ResNet when using them. The AMID/HCF design is related to
*Multi-layer Denoising Fusion Model for Multimodal Aspect-Based Sentiment Analysis* (MDFM);
retain appropriate attribution when describing this configuration.

## Validation status

This release preserves the original training script's model operations and evaluation rules
while parameterizing input/output paths and adding input validation. Full training on the
private inputs has not been rerun as part of preparing this source-only release.
