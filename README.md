# Python NLP testing app

### Setup
Install virtual environment:

```
python3 -m venv --system-site-packages ./venv
```

Start virtual environment:

```
source ./venv/bin/activate
```

Install dependencies:

```
pip3 install -e .
```

If you get warnings about upgrading pip or setuptools:

```
pip3 install --upgrade pip setuptools
```

Run the program:

```
venv/bin/python3 src/nlp/sentiment/nlp.sa.py
```

### Resources
Find other pre-trained models:  
https://huggingface.co/models?filter=text-classification&search=xlnet

Pretraining docs:  
https://towardsdatascience.com/multi-class-sentiment-analysis-using-bert-86657a2af156
