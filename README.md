# Neural Machine Translation by Jointly Learning to Align and Translate

This part of the repo contains my paper implementation and training code for german to english translation using the **RNNsearch** model. If you find any mistakes or stuff that I've could build better, please let me know.

## Training

Before you start training you should make sure, that you have installed all the packages from the `requirements.txt` and  downloaded the spacy tokenizers:
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```
If you want to use this for other languages you have to download the tokenizers you need and maybe make some minor changes in the `config.py` and `dataset.py`.

You find the custom configurations in `config.py`. If you want to change some hyperparameters you can create your own `custom_config.yaml` containing all hyperparameters you want to override.

```
python train.py --config="your_custom_config.yaml"
```

Currently I haven't implemented mixed precision of multi-GPU training.

## Pretrained Model

I've trained a small model for 36 epochs so you can use it to test my code. You can find the model and the corresponding config file [here](https://drive.google.com/drive/folders/17tyZX-GtC5YQpiyEqXm2DkIhRVldKX4h?usp=sharing).

## References

Here are some references I used:

* [Original Paper](https://arxiv.org/pdf/1409.0473.pdf)
* [Notebooks](https://github.com/bentrevett/pytorch-seq2seq)