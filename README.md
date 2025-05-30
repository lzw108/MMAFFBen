# MMAFFBen: A Multilingual and Multimodal Affective Analysis Benchmark for Evaluating LLMs and VLMs

This is an extensive open-source benchmark for multilingual multimodal affective analysis. 

Paper link: [MMAFFBen](https://github.com/lzw108/MMAFFBen)

## Datasets

- [MMAFFBen](https://huggingface.co/datasets/lzw1008/MMAFFBen)
- [MMAFFIn](https://huggingface.co/datasets/lzw1008/MMAFFIn)

## Model

- [MMAFFLM-3b](https://huggingface.co/lzw1008/MMAFFLM-3b)
- [MMAFFLM-7b](https://huggingface.co/lzw1008/MMAFFLM-7b)

## Usage

### Fine-tune your model based on MMAFFIn

Download the train datasets (MMAFFIn) to the data folder.

```python
bash run_sft_stream.sh
```

### Evaluate your model on MMAFFBen

Download MMAFFBen data to the data folder.

```python
bash run_inference.sh
```
This code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The current version supports the Qwen-VL series. Adjust the code for your own model according to the guidelines according to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

After getting the predicted results in the predicts folder, follow the steps in the evaluation.ipynb to obtain the scores of each subdataset.



