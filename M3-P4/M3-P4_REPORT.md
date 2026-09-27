# M3-P4 — Neural Speech Enhancement

## 1. Objective

M3-P4 investigates a lightweight neural speech-enhancement system for noisy speech. The neural enhancement stage is evaluated as a conditional component: it should only be considered for the final system if it provides meaningful enhancement while maintaining acceptable computational cost and latency.

## 2. Dataset

The experiment uses the prepared M3-P2 speech/noise mixture dataset.

- Speech WAV files: 219
- Noise WAV files: 2000
- Mixture WAV files: 800
- Mixture sample rate: 16 kHz
- Speech source: LibriSpeech train-clean-100 subset
- Noise source: ESC-50
- Speech files used for mixtures: 200
- SNR levels: -5, 0, +5, +10 dB
- Random seed: 42

The 800 mixtures contain 200 examples at each SNR level.

## 3. Preprocessing

Short-Time Fourier Transform (STFT) preprocessing was implemented.

Parameters:

- Sampling rate: 16000 Hz
- FFT size: 512
- Window size: 512 samples
- Hop size: 128 samples
- Frequency bins: 257
- Input representation: magnitude
- Input transformation: log1p magnitude

The STFT/ISTFT reconstruction was validated with a maximum reconstruction error of approximately 8.94e-08 samples on the validation example.

## 4. Neural Enhancement Architecture

A lightweight convolutional neural network (CNN) was selected for the initial neural enhancement experiment.

The network receives a single-channel spectral representation:

    [batch, channel, frequency, time]

and produces an output mask with the same dimensions.

Example model test:

- Input shape: (1, 1, 257, 100)
- Output shape: (1, 1, 257, 100)
- Number of parameters: 18,817

The model is therefore substantially smaller than large transformer-style speech enhancement architectures and is suitable for CPU experimentation.

## 5. Enhancement Target

The model predicts an Ideal Ratio Mask (IRM).

The target mask is computed from the clean speech magnitude and mixture magnitude:

    IRM = speech_magnitude / (mixture_magnitude + epsilon)

The mask is clipped to the range:

    [0, 1]

The predicted mask is applied to the mixture magnitude before reconstruction.

## 6. Training Configuration

The training configuration was:

- Model: CNN
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 4
- Epochs: 5
- Train split: 80%
- Validation split: 20%
- Random seed: 42
- Device: CPU

Dataset split:

- Total samples: 800
- Training samples: 640
- Validation samples: 160

## 7. Training Results

Training was completed successfully.

| Epoch | Training Loss | Validation Loss |
|------:|--------------:|----------------:|
| 1 | 0.149296 | 0.148703 |
| 2 | 0.144086 | 0.147137 |
| 3 | 0.139583 | 0.138338 |
| 4 | 0.130992 | 0.129633 |
| 5 | 0.126511 | 0.124557 |

The validation loss decreased from 0.148703 at epoch 1 to 0.124557 at epoch 5.

## 8. Trained Model

The trained model was saved as:

    M3-P4/model/trained_model.pth

Training metrics were saved as:

    M3-P4/model/training_metrics.csv

Model parameter count:

    18,817

## 9. Evaluation

The trained CNN was evaluated on 20 mixture samples covering all four tested SNR levels.

Evaluation output:

- Samples evaluated: 20
- Average SNR improvement: 3.1967 dB
- Median SNR improvement: 2.7446 dB
- Average inference time: 0.115376 seconds
- Device: CPU

Enhanced audio files were generated for all 20 evaluated mixtures.

Evaluation results were saved to:

    M3-P4/evaluation/evaluation_results.csv

Enhanced audio was saved under:

    M3-P4/evaluation/enhanced_audio/

## 10. SNR Improvement Results

The measured SNR improvement varied according to the input condition.

Observed improvements in the evaluated samples ranged from approximately:

    -0.790 dB to +9.400 dB

The results therefore show that the neural enhancement stage can provide measurable improvement for many noisy inputs, while performance is not uniformly positive for every sample.

Average improvement:

    +3.1967 dB

Median improvement:

    +2.7446 dB

## 11. Inference Performance

Average inference time for the evaluated samples was:

    0.115376 seconds

The implementation was executed on CPU.

The model contains only 18,817 trainable parameters, supporting the objective of investigating a lightweight neural enhancement approach.

## 12. Reproducibility

The experiment uses:

- Random seed: 42
- Fixed STFT parameters
- Fixed dataset metadata
- Fixed train/validation split configuration
- Fixed CNN architecture
- Fixed optimizer and learning rate
- Fixed batch size
- Fixed number of epochs

The training configuration is also stored in:

    M3-P4/training/configs/cnn_config.yaml

## 13. Deliverables

The M3-P4 implementation contains:

    M3-P4/
    ├── evaluation/
    │   ├── evaluate.py
    │   ├── evaluation_results.csv
    │   └── enhanced_audio/
    │
    ├── model/
    │   ├── trained_model.pth
    │   └── training_metrics.csv
    │
    ├── models/
    │   ├── cnn/
    │   │   └── model.py
    │   └── crnn/
    │
    ├── preprocessing/
    │   └── stft.py
    │
    ├── training/
    │   ├── dataset.py
    │   ├── train.py
    │   └── configs/
    │       └── cnn_config.yaml
    │
    └── M3-P4_REPORT.md

## 14. Limitations

The evaluation was performed on 20 samples from the 800 available mixtures.

The measured SNR improvement is therefore an evaluation result for this test subset and should not be interpreted as a complete characterization of all 800 mixtures.

The inference measurements were obtained on a CPU system.

Some evaluated samples showed negative SNR improvement, indicating that the enhancement model does not improve every input condition.

The current implementation investigates the CNN candidate. A CRNN implementation was not trained or evaluated in this experiment.

## 15. Conditional Integration Assessment

The purpose of M3-P4 is to determine whether neural enhancement provides sufficient practical benefit to justify inclusion in the final system.

The completed experiment demonstrates:

- A small CNN with 18,817 parameters
- CPU-compatible inference
- Successful reproducible training
- Average measured SNR improvement of 3.1967 dB on the evaluated subset
- Average inference time of 0.115376 seconds
- Non-uniform performance across different input conditions

These measurements provide the experimental evidence needed for comparison with the existing ANC pipeline. Final system integration should be based on the project's broader latency, stability, and enhancement requirements rather than on SNR improvement alone.

## 16. Conclusion

M3-P4 successfully implemented and evaluated a lightweight CNN-based neural speech-enhancement pipeline.

The complete pipeline includes:

    Noisy Audio
        ↓
       STFT
        ↓
    Magnitude Representation
        ↓
    Lightweight CNN
        ↓
    Predicted Mask
        ↓
    Enhanced Spectrum
        ↓
       ISTFT
        ↓
    Enhanced Audio

The model was trained successfully, saved, and evaluated on CPU. The resulting measurements provide a reproducible baseline for determining the role of neural speech enhancement in the final ANC system.
