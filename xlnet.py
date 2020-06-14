import tensorflow
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline


def readData(filename, f):
    with open(filename) as file:
        for line in file: f(line)


def setupDefaultSentimentAnalysis() -> TextClassificationPipeline:
    return pipeline(task="sentiment-analysis")


def setupXLNetSentimentAnalysis(modelName):
    tokenizer = XLNetTokenizer.from_pretrained(modelName)
    model = XLNetForSequenceClassification.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)


def setupBertSentimentAnalysis(modelName):
    tokenizer = BertTokenizer.from_pretrained(modelName)
    model = BertForSequenceClassification.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

def setupAuto(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

def setup(model="default pipline"):
    print(model, ':')
    return {
        "default": lambda: setupAuto("distilbert-base-uncased-finetuned-sst-2-english"),
        "xlnet": lambda: setupXLNetSentimentAnalysis("textattack/xlnet-large-cased-SST-2"),
        "bert": lambda: setupBertSentimentAnalysis("bert-base-uncased"),
        "imdb": lambda: setupAuto("lvwerra/bert-imdb"),
        "finbert": lambda: setupAuto("ipuneetrathore/bert-base-cased-finetuned-finBERT")
    }.get(model, lambda: setupDefaultSentimentAnalysis())()

def main():

    def newline(): print()
    def result(f, j, s:str): f("Result: ", j(s))

    negativeStr = "This film is terrible."
    positiveStr = "This film is great."
    neutralStr = "I saw the film."

    print("Versions:")
    print("Torch ", torch.__version__)
    print("Tensorflow ", tensorflow.__version__)
    print("Transformers ", transformers.__version__)
    newline()

    models = [
        "default", "bert", "imdb", "finbert", "xlnet"
    ]

    for model in models:
        nlp = setup(model)
        result(print, nlp, negativeStr)
        result(print, nlp, positiveStr)
        result(print, nlp, neutralStr)
        newline()


if __name__ == "__main__":
    main()
