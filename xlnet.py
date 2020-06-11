import logging

from transformers import pipeline

# logging config doc: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
logging.basicConfig(level="WARN", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    nlp = pipeline("sentiment-analysis")
    result = nlp("We are very happy to include pipeline into the transformers repository.")
    logger.warning("Result: %s %s", result[0]['label'], result[0]['score'])

if __name__ == "__main__":
    main()
