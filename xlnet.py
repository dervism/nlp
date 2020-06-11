from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

def main():
    # Allocate a pipeline for sentiment-analysis
    nlp = pipeline("sentiment-analysis")
    result = nlp("We are very happy to include pipeline into the transformers repository.")
    print(result);

if __name__ == "__main__":
    main()
