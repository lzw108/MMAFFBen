# MMAFFBen: A Multilingual and Multimodal Affective Analysis Benchmark

This is an extensive open-source benchmark for multilingual multimodal affective analysis. 

Paper link: [MMAFFBen](https://github.com/lzw108/MMAFFBen/edit/main/README.md)

## Datasets

- [MMAFFBen](https://huggingface.co/datasets/lzw1008/MMAFFBen)
- [MMAFFIn](https://huggingface.co/datasets/lzw1008/MMAFFIn)

## Model

- [MMAFFLM-3b]()
- [MMAFFLM-7b]()

## Usage

This code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Fine-tune your model based on MMAFFIn

Download the train datasets to the data folder.

```python
bash run_sft_stream.sh
```
