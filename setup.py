from setuptools import setup, find_namespace_packages

from clean import CleanCommand

cmd_classes = {}
cmd_classes['clean'] = CleanCommand

setup(
    name='sentiment-demo',
    version='0.0.1',
    packages=find_namespace_packages(where="src", include=["nlp.*"]),
    install_requires=['transformers', 'torch', 'torchvision', 'tensorflow', 'openpyxl', 'stanza'],

    entry_points={
        "console_scripts": [
            "models = nlp.sentiment:testModels",
            "dataset = nlp.sentiment:testDataset"
        ]
    },

    package_dir={"": "src"},

    package_data={
        "nlp.sentiment": ["data/*.txt"],
    },

    cmdclass=cmd_classes,

    url='https://github.com/dervism/nlp',
    license='MIT',
    author='dervism',
    author_email='dervis4@gmail.com',
    description='NLP Sentiment Analysis test'
)
