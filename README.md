

Install virtual environment:

```
python3 -m venv --system-site-packages ./venv
```


Start virtual environment:

```
source ./venv/bin/activate
```

Install upgrades:

```
pip install --upgrade pip setuptools
```

Install Tensorflow:

```
pip install --upgrade tensorflow
```


Verify Tensorflow install:

```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Install PyTorch:

```
pip3 install torch torchvision
```

Verify PyTorch install:

```
python -c "import torch;print(torch.rand(5, 3))"
```

Run the program:

```
python nlp.sa.transformers.py
```

Find other pre-trained models:

https://huggingface.co/models?filter=text-classification&search=xlnet

Pretraining docs:
https://towardsdatascience.com/multi-class-sentiment-analysis-using-bert-86657a2af156
