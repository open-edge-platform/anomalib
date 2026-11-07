# Dinomaly: Vision Transformer-based Anomaly Detection with Feature Reconstruction

This is the implementation of the Dinomaly model based on the [original implementation](https://github.com/guojiajeremy/Dinomaly).

Model Type: Segmentation

## Description

Dinomaly is a Vision Transformer-based anomaly detection model that uses an encoder-decoder architecture
for feature reconstruction.
The model leverages pre-trained DINOv2 Vision Transformer features and employs a reconstruction-based approach
to detect anomalies by comparing encoder and decoder features.

### Architecture

The Dinomaly model consists of three main components:

1. DINOv2 Encoder: A pre-trained Vision Transformer (ViT) which extracts multi-scale feature maps.
2. Bottleneck MLP: A simple feed-forward network that collects features from the encoder's middle layers
   (e.g., 8 out of 12 layers for ViT-Base).
3. Vision Transformer Decoder: Consisting of Transformer layers (typically 8), it learns to reconstruct the
   compressed middle-level features by maximising cosine similarity with the encoder's features.

Only the parameters of the bottleneck MLP and the decoder are trained.

#### Key Components

1. Foundation Transformer Models: Dinomaly leverages pre-trained ViTs (like DinoV2) which provide universal and
   discriminative features. This use of foundation models enables strong performance across various image patterns.
2. Noisy Bottleneck: This component activates built-in Dropout within the MLP bottleneck.
   By randomly discarding neural activations, Dropout acts as a "pseudo feature anomaly," which forces the decoder
   to restore only normal features. This helps prevent the decoder from becoming too adept at reconstructing
   anomalous patterns it has not been specifically trained on.
3. Linear Attention: Instead of traditional Softmax Attention, Linear Attention is used in the decoder.
   Linear Attention's inherent inability to heavily focus on local regions, a characteristic sometimes seen as a
   "side effect" in supervised tasks, is exploited here. This property encourages attention to spread across
   the entire image, reducing the likelihood of the decoder simply forwarding identical information
   from unexpected or anomalous patterns. This also contributes to computational efficiency.
4. Loose Reconstruction:
   1. Loose Constraint: Rather than enforcing rigid layer-to-layer reconstruction, Dinomaly groups multiple
      encoder layers as a whole for reconstruction (e.g., into low-semantic and high-semantic groups).
      This provides the decoder with more degrees of freedom, allowing it to behave more distinctly from the
      encoder when encountering unseen patterns.
   2. Loose Loss: The point-by-point reconstruction loss function is loosened by employing a hard-mining
      global cosine loss. This approach detaches the gradients of feature points that are already well-reconstructed
      during training, preventing the model from becoming overly proficient at reconstructing all features,
      including those that might correspond to anomalies.

### Anomaly Detection

Anomaly detection is performed by computing cosine similarity between encoder and decoder features at multiple scales.
The model generates anomaly maps by analyzing the reconstruction quality of features, where poor reconstruction
indicates anomalous regions. Both anomaly detection (image-level) and localization (pixel-level) are supported.

## Usage

`anomalib train --model Dinomaly --data MVTecAD --data.category <category>`

## Benchmark

All results gathered with seed `42`. The `max_steps` parameter is set to `5000` for training and
batch size is set to `16.'

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|          |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| -------- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| Dinomaly | 0.997 | 1.000  | 1.000 |  0.986  | 0.999  | 0.999 |  1.000   |  1.000  |   1.000   | 0.995 | 0.986 | 1.000 |   1.000    |   0.997    | 0.996 | 1.000  |

### Image F1 Score

|          |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| -------- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| Dinomaly | 0.987 | 1.000  | 1.000 |  0.982  | 0.994  | 0.991 |  0.993   |  0.995  |   0.984   | 0.979 | 0.961 | 1.000 |   0.983    |   0.963    | 0.984 | 0.996  |

### Pixel-Level AUC

|          |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| -------- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| Dinomaly | 0.984 | 0.990  | 0.985 |  0.986  | 0.993  | 0.994 |  0.994   |  0.992  |   0.966   | 0.980 | 0.997 | 0.976 |   0.989    |   0.951    | 0.977 | 0.991  |

### Pixel F1 Score

|          |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| -------- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| Dinomaly | 0.670 | 0.822  | 0.732 |  0.568  | 0.700  | 0.556 |  0.756   |  0.467  |   0.851   | 0.687 | 0.572 | 0.736 |   0.608    |   0.615    | 0.670 | 0.716  |
