import argparse
import logging
from dataclasses import dataclass
from typing import List

import tensorflow
import torch
import transformers
from openpyxl import Workbook
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BartTokenizer, BartForSequenceClassification, AutoModelWithLMHead
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline

# import stanza


# root logging config doc
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# app logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


@dataclass
class SentimentResult:
    sentiment: str
    confidence: float
    text: str

@dataclass
class SentimentResultList:
    sentiments: List[SentimentResult]

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

def setupBartSentimentAnalysis(modelName):
    tokenizer = BartTokenizer.from_pretrained(modelName)
    model = BartForSequenceClassification.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

def setupBart2SentimentAnalysis(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelWithLMHead.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

def setupAuto(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

def setupStanza():
    #stanza.download('en')
    #nlp = stanza.Pipeline('en')
    #result = nlp(text)
    print("Stanza doesn't support SA yet.")

def setup(model="default pipline"):
    print(model, ':')
    return {
        "xlnet": lambda: setupXLNetSentimentAnalysis("textattack/xlnet-large-cased-SST-2"),
        "xlnetbase": lambda: setupAuto("textattack/xlnet-base-cased-SST-2"),
        "bertbase": lambda: setupBertSentimentAnalysis("textattack/bert-base-uncased-SST-2"),
        "finance": lambda: setupAuto("ipuneetrathore/bert-base-cased-finetuned-finBERT"),
        "bart": lambda: setupBartSentimentAnalysis("textattack/facebook-bart-large-SST-2"),
        "stanza": lambda: setupStanza(),
    }.get(model, lambda: setupDefaultSentimentAnalysis())()

def result2Str(model, result):
    return label2Str(model, result) + " " + score2Str(result)

def label2Str(model, result):
    label = result[0]['label']
    model = "generic" if "xlnet" or "bert" in label else model
    labelStr = {
        "generic": {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "POSITIVE"
        }.get(label, label),
        "finance": {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }.get(label, label),
        "generic_finetuned" : {
            "LABEL_0": "LABEL_0",
            "LABEL_1": "LABEL_1",
            "LABEL_2": "LABEL_2",
            "LABEL_3": "LABEL_3",
            "LABEL_4": "LABEL_4"
        }.get(label, label),
    }.get(model, label)

    return labelStr

def score2Str(result):
    return "{:.2%}".format(result[0]['score'])


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
        "default", "xlnet", "xlnetbase", "bertbase"
    ]

    for model in models:
        nlp = setup(model)
        print("Result: ", result2Str(model, nlp(negativeStr)))
        print("Result: ", result2Str(model, nlp(positiveStr)))
        print("Result: ", result2Str(model, nlp(neutralStr)))
        print()


def testDataset(model):
    workbook = Workbook()
    sheet = workbook.active
    sentiments = []

    nlp = setup(model)
    with open(outputCSVFile, "w") as output:
        with open(inputFile, "r") as file:
            for line in file:
                print(line)
                result = nlp(line[1:-1])
                output.write(line.strip() + "\t" + result2Str(model, result))
                output.write("\n")
                sentimentResult = SentimentResult(
                    sentiment=label2Str(model, result),
                    confidence=float(result[0]['score']),
                    text=line
                )
                sentiments.append(sentimentResult)

        sentimentResultList = SentimentResultList(sentiments=sentiments)
        sheet.append(["Text", "Sentiment", "Confidence"])

        for sentimentObj in sentimentResultList.sentiments:
            data = [sentimentObj.text, sentimentObj.sentiment, sentimentObj.confidence]
            sheet.append(data)
        workbook.save(filename=outputFolder + outputExcelFile)

outputFolder: str = ""
def setOutputFolder(path):
    global outputFolder
    outputFolder = path

outputExcelFile: str = "output.xlsx"
def setOutputExcelFile(name):
    global outputExcelFile
    outputExcelFile = name

outputCSVFile: str = "output.csv"
def setOutputCSVFile(name):
    global outputCSVFile
    outputCSVFile = name

inputFile: str = "./data/test.txt"
def setInputFile(name):
    global inputFile
    inputFile = name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XLNet Sentiment Analsis')
    parser.add_argument('--model', type=str, default='xlnetbase')
    args = parser.parse_args()
    testModels()