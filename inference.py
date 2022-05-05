import argparse
import torch
import youtokentome
import os
from nltk import tokenize


from models import *

def inference(model, text, device, top_k=None, threshold=0):
    """
    Generate the extractive summary
    """
    model.eval()
    logits = model(text['inputs'])
    if top_k:
        sum_in = torch.argsort(logits, dim=1)[:, -top_k:] 
    else:
        sum_in = (logits > threshold).nonzero(as_tuple=False)
        
    pred_summary = ' '.join([text['records'][ind] for ind in sum_in.sort(dim=1)[0][0] if ind < len(text["records"])]) 
    return pred_summary

def main(model_path, tokenizer_path, input_file_path):
    max_sentences = 30
    max_sentence_length = 30
    lower = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = youtokentome.BPE(tokenizer_path)
    vocabulary = tokenizer.vocab()
    vocab_size = len(vocabulary)
    model = SentenceTaggerRNN(vocab_size).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    if os.path.isfile(input_file_path):
        file = open(input_file_path,'r')
        data = file.read() 
        assert len(data) > 1, "File is empty"
        sentences = [sentence.lower() if lower else sentence for sentence in tokenize.sent_tokenize(data)][:max_sentences]
        inputs = [tokenizer.encode(sentence)[:max_sentence_length] for sentence in sentences]

        max_sentence_length = max(max_sentence_length, max([len(tokens) for tokens in inputs]))
        tensor_inputs = torch.zeros((max_sentences, max_sentence_length), dtype=torch.long, device=device)
                  
        for i, inputs in enumerate(inputs):
            for j, sentence_tokens in enumerate(inputs):
                tensor_inputs[i][j] = torch.tensor(sentence_tokens, dtype=torch.int64)

                txt = {'inputs': tensor_inputs.unsqueeze(0),
                      'records': sentences
                      }
        prediction = inference(model, txt, device, top_k=3, threshold=0)
        output = open('./output.txt','w+')
        output.write(prediction)
        print('File written')
    else:
        print('Input file does not exist.')

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model.pt', required=True, help='Path to the model weights')
    parser.add_argument('--tokenizer_path', type=str, default='./BPE_model.bin', required=True, help='BPE tokenizer path')
    parser.add_argument('--input_file_path', type=str, required=True, help='Input file path')

    args = parser.parse_args()
    main(**vars(args))
    