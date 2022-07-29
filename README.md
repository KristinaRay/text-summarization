# Extractive Text Summarization

This is a PyTorch implementation of the paper [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/pdf/1611.04230.pdf)


## Demo

Clone the project using the command line

```
git clone https://github.com/KristinaRay/text-summarization.git
```
```
cd text-summarization
``` 

To start Flask server locally run

```
python demo/app.py --model_dir <model_path> \
                   --tokenizer_dir <BPE_model_path> \
                   --host <host> \
                   --port <port>
```

## Dependencies

Install dependencies by running

```
pip install -r requirements.txt
```

## Dataset

CNN/DailyMail non-anonymized summarization dataset is used for training the extractive summarization model.

In order to get the dataset, run

```
python fetch_dataset.py
```

For more information about CNN/DailyMail dataset, please follow the [link](https://www.tensorflow.org/datasets/catalog/cnn_dailymail)

## Usage

To train the extractive summarization model, run

```
python main.py --num_samples <number of dataset samples> \
                --batch_size <batch size> \
                --num_epochs <number of epochs> \
                --clip <gradient clip value> \
                --seed <a seed value for random number generator>
```

Generate the extractive summary of a text file

```
python inference.py --model_path <path to the trained model> \
                --tokenizer_path <path to the trained BPE tokenizer> \
                --input_file_path <path to a text file>
```

## Output

The summary of a text file is saved in txt file  ```./output.txt```
