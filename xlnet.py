import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import pipeline


def setupDefaultSentimentAnalysis() -> TextClassificationPipeline:
    return pipeline(task="sentiment-analysis")


def setupSentimentAnalysis(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    return (tokenizer, model, pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer))


def main():
    parser = argparse.ArgumentParser(description='XLNet Language Model')
    parser.add_argument('--model', type=str, default='xlnet')
    args = parser.parse_args()

    negativeStr = "This film is terrible"
    positiveStr = "This film is great"

    _, _, nlp = setupSentimentAnalysis("bert-base-cased")
    print("bert-base-cased:")
    print("Result: %s", nlp(negativeStr))
    print("Result: %s", nlp(positiveStr))
    print()

    tokenizer, model, nlp = setupSentimentAnalysis("textattack/xlnet-large-cased-STS-B")
    print("textattack/xlnet-large-cased-STS-B:")
    print("Result: %s", nlp(negativeStr))
    print("Result: %s", nlp(positiveStr))
    print()

    # Example is from https://huggingface.co/transformers/model_doc/xlnet.html#xlnetforsequenceclassification
    print("Calling the model directly (HuggingFace example from the documentation)")
    input_ids = torch.tensor(tokenizer.encode(negativeStr, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids, labels=torch.tensor([1]).unsqueeze(0))
    print("Result %s", outputs)

    input_ids = torch.tensor(tokenizer.encode(positiveStr, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids, labels=torch.tensor([1]).unsqueeze(0))
    print("Result %s", outputs)


if __name__ == "__main__":
    main()
