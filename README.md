---
language:
- kk
metrics:
- wer
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- speech
- audio
- pytorch
- stt
---


## Model Overview

In order to prepare and experiment with the model, it's necessary to install [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo) [1].\
\
This model have been trained on NVIDIA GeForce RTX 2070:\
Python 3.7.15\
NumPy 1.21.6\
PyTorch 1.21.1\
NVIDIA NeMo 1.7.0

```bash
pip3 install nemo_toolkit['all']
```

### Model Usage:

The model is accessible within the NeMo toolkit [1] and can serve as a pre-trained checkpoint for either making inferences or for fine-tuning on a different dataset.

### How to Import

```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.restore_from(restore_path="stt_kz_quartznet15x5.nemo")
```

### How to Train

```bash
python3 train.py \
--train_manifest path/to/manifest.json \
--val_manifest path/to/manifest.json \
--accelerator "gpu" \
--batch_size BATCH_SIZE \
--num_epochs NUM_EPOCHS \
--model_save_path path/to/save/model.nemo
```

### How to Evaluate

```bash
python3 evaluate.py \
--model_path /path/to/model.nemo \
--test_manifest path/to/manifest.json \
--batch_size BATCH_SIZE
```

### How to Transcribe Audio File

Sample audio to test the model:
```bash
wget https://asr-kz-example.s3.us-west-2.amazonaws.com/sample_kz.wav
```
This line is to transcribe the single audio:
```bash
python3 transcribe.py --model_path /path/to/model.nemo --audio_file_path path/to/audio/file
```

### Input and Output

This model can take input from mono-channel audio .WAV files with a sample rate of 16,000 KHz.\
Then, this model gives you the spoken words in a text format for a given audio sample.

### Model Architecture

[QuartzNet 15x5](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5) [2] is a Jasper-like network that uses separable convolutions and larger filter sizes. It has comparable accuracy to Jasper while having much fewer parameters. This particular model has 15 blocks each repeated 5 times.

### Training and Dataset

The model was finetuned to Kazakh speech based on the pre-trained English Model for over several epochs.
[Kazakh Speech Corpus 2](https://issai.nu.edu.kz/kz-speech-corpus/?version=1.1) (KSC2) [3] is the first industrial-scale open-source Kazakh speech corpus.\
In total, KSC2 contains around 1.2k hours of high-quality transcribed data comprising over 600k utterances.

### Performance
The model achieved:\
Average WER: 13.53%\
through the applying of **Greedy Decoding**.

### Limitations

Because the GPU has limited power, lightweight model architecture was used for fine-tuning.\
In general, this makes it faster for inference but might show less overall performance.\
In addition, if the speech includes technical terms or dialect words the model hasn't learned, it may not work as well.

### Demonstration

For inference and downloading the model, check on Hugging Face Space: [NeMo_STT_KZ_Quartznet15x5](https://huggingface.co/spaces/transiteration/nemo_stt_kz_quartznet15x5)

### References

[1] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[2] [QuartzNet 15x5](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5)

[3] [Kazakh Speech Corpus 2](https://issai.nu.edu.kz/kz-speech-corpus/?version=1.1)
