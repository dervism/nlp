import tensorflow
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline


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

def result2Str(model, result):
    label = result[0]['label']
    score = result[0]['score']
    labelStr = {
        "finbert": {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }.get(label, label)
    }.get(model, label)

    return labelStr + " {:.2%}".format(score)

def testModels():

    negativeStr = "This film is terrible."
    positiveStr = "This film is great."
    neutralStr = "I saw the film."

    print("Versions:")
    print("Torch ", torch.__version__)
    print("Tensorflow ", tensorflow.__version__)
    print("Transformers ", transformers.__version__)
    print()

    models = [
        "default", "finbert", "imdb", "bert", "xlnet"
    ]

    for model in models:
        nlp = setup(model)
        print("Result: ", result2Str(model, nlp(negativeStr)))
        print("Result: ", result2Str(model, nlp(positiveStr)))
        print("Result: ", result2Str(model, nlp(neutralStr)))
        print()


def testDataset():
    nlp = setup("imdb")
    with open("./data/output.csv", "w") as output:
        with open("./data/test.txt") as file:
            for line in file:
                print(line)
                result = nlp(line[1:-1])
                label = {
                    "LABEL_0": "NEGATIVE",
                    "LABEL_1": "POSITIVE"
                }.get(result[0]['label'], result[0]['label'])
                output.write(line.strip() + "\t" + label)
                output.write("\n")

if __name__ == "__main__":
    testModels()
