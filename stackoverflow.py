import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline


def setupDefaultSentimentAnalysis() -> TextClassificationPipeline:
    return pipeline(task="sentiment-analysis")


def setupXLNetSentimentAnalysis(modelName):
    tokenizer = XLNetTokenizer.from_pretrained(modelName)
    model = XLNetForSequenceClassification.from_pretrained(modelName)
    return (tokenizer, model, pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer))


def main():

    negativeStr = "This film is terrible."
    positiveStr = "This film is great."

    nlp = pipeline(task="sentiment-analysis")
    print("Result: ", nlp(negativeStr))
    print("Result: ", nlp(positiveStr))
    print()

    tokenizer = AutoTokenizer.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
    model = AutoModelForSequenceClassification.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
    nlp = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Result: ", nlp(negativeStr))
    print("Result: ", nlp(positiveStr))
    print()

    tokenizer = XLNetTokenizer.from_pretrained("textattack/xlnet-large-cased-STS-B")
    model = XLNetForSequenceClassification.from_pretrained("textattack/xlnet-large-cased-STS-B")
    nlp = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Result: ", nlp(negativeStr))
    print("Result: ", nlp(positiveStr))
    print()

    # Example is from https://huggingface.co/transformers/model_doc/xlnet.html#xlnetforsequenceclassification
    print("Calling the model directly (HuggingFace example from the documentation)")
    input_ids = torch.tensor(tokenizer.encode(negativeStr, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids, labels=torch.tensor([1]).unsqueeze(0))
    print("Result ", outputs)


if __name__ == "__main__":
    main()
