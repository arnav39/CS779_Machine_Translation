import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import numpy as np
import spacy
from tqdm import tqdm
import sys
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
import pickle
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"device = {device}")

# hyper-params

MAX_LENGTH = 64
D_MODEL = 200
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
BATCH_SIZE = 32


# classes

def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class Lang():

  def __init__(self, name, spacy_tokenizer):
    self.name = name
    self.word2index = {"<SOS>":0, '<EOS>': 1, "<PAD>": 2, '<UNK>': 3}
    self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: '<UNK>'}
    self.word2count = {}
    self.n_words = 4
    self.tokenizer = spacy_tokenizer

  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1

    else:
      self.word2count[word] += 1

  def add_sentence(self, sentence):
    tokens = self.tokenize_sentence(sentence)
    for token in tokens: 
      self.add_word(token)

  def tokenize_sentence(self, sentence):
    tokens = [token.text for token in self.tokenizer(sentence.lower())]
    return tokens

  def __len__(self):
    return self.n_words

class TestSet(Dataset):

    def __init__(self, test_sent_tensor_list):
        super().__init__()
        self.inp = test_sent_tensor_list

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, index):
        return self.inp[index]

class Hindi_lang():

  def __init__(self, name):
    self.name = name
    self.word2index = {"<SOS>":0, '<EOS>': 1, "<PAD>": 2, '<UNK>': 3}
    self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: '<UNK>'}
    self.word2count = {}
    self.n_words = 4
    self.normalizer = indic_normalize.DevanagariNormalizer(lang='hi', remove_nuktas=True)

  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1

    else:
      self.word2count[word] += 1

  def add_sentence(self, sentence):
    tokens = self.tokenize_sentence(sentence)
    for token in tokens: 
      self.add_word(token)

  def tokenize_sentence(self, sentence):
    # first normalize the sentence, then tokenize
    norm_sent = self.normalizer.normalize(sentence)
    tokens = indic_tokenize.trivial_tokenize(norm_sent)
    return tokens

  def __len__(self):
    return self.n_words

with open('new/hindi_input_vocab.pkl', "rb") as f:
    hindi_input_vocab = pickle.load(f)

with open('new/english_output_vocab.pkl', 'rb') as f: 
    english_output_vocab = pickle.load(f)

with open('new/test_loader.pkl', 'rb') as f:
  test_loader = pickle.load(f)

print(len(hindi_input_vocab))
print(len(english_output_vocab))

SOS_TOKEN_INDEX = english_output_vocab.word2index['<SOS>']
EOS_TOKEN_INDEX = english_output_vocab.word2index['<EOS>']
PAD_TOKEN_INDEX = english_output_vocab.word2index['<PAD>']

class Transformer(nn.Module):

  def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):

    super().__init__()
    self.input_vocab_size = input_vocab_size
    self.output_vocab_size = output_vocab_size
    self.d_model = d_model
    self.nhead = nhead 
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers
    self.dim_feedforward = dim_feedforward
    self.dropout = dropout

    self.input_embedding = nn.Embedding(input_vocab_size, d_model)
    transformer_enc_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, batch_first=True)
    self.encoder = nn.TransformerEncoder(transformer_enc_layer, self.num_encoder_layers)

    self.output_embedding = nn.Embedding(output_vocab_size, d_model)
    transformer_dec_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, batch_first=True)
    self.decoder = nn.TransformerDecoder(transformer_dec_layer, self.num_decoder_layers)

    self.out = nn.Linear(d_model, output_vocab_size)

  def forward(self, src, trg):

    # src.shape --> (batch_size, seq_len)
    # trg.shape --> (batch_size, seq_len)

    # output is a tensor of shape : (batch_size, trg_seq_len, output_vocab_size)

    src_mask = self.generate_src_mask(src.size(1), trg.size(1)).to(device) # (trg_seq_len, src_seq_len)
    trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(device) # (tgt_seq_len, tgt_seq_len)

    src_embed = self.input_embedding(src).to(device) # (batch_size, seq_len, embed_dim)
    src_enc = self.encoder(src_embed).to(device) # (batch_size, seq_len, embed_dim)

    src_padding_mask = self.generate_padding_mask(src).to(device) # (batch_size, seq_len)
    trg_padding_mask = self.generate_padding_mask(trg).to(device) # (batch_size, seq_len)

    trg_embed = self.output_embedding(trg).to(device) # (batch_size seq_len, embed_dim)
    trg_dec = self.decoder.forward(tgt=trg_embed, memory=src_embed, tgt_mask=trg_mask, memory_mask=src_mask, tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=src_padding_mask)

    # trg_dec.shape --> (batch_size, tgt_seq_len, embed_dim)

    output = self.out(trg_dec.to(device)) # (batch_size, tgt_seq_len, output_vocab_size)
    return output

  def generate_src_mask(self, src_length, trg_length):

    mask = torch.zeros(trg_length, src_length)
    for i in range(trg_length):
      mask[i, :i+1] = 1

    mask = mask.masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

  def generate_square_subsequent_mask(self, sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

  def generate_padding_mask(self, my_tensor):

    # my_tensor.shape --> (batch_size, seq_len)
    # returns the padding mask tensor of shape (batch_size, seq_len)

    PADDING_INDEX = hindi_input_vocab.word2index['<PAD>']

    padding_lists = []
    for my_sent in my_tensor:
      pad_list = []
      for word in my_sent:
        if word == PADDING_INDEX:
          pad_list.append(0)
        else:
          pad_list.append(1)

      padding_lists.append(pad_list)

    padding_tensor = torch.tensor(padding_lists, dtype=torch.float32)
    return padding_tensor

@add_to_class(Transformer)
def forward_eval(self, src: torch.Tensor):

  # src.shape --> (batch_size, seq_len)

    batch_size = src.size(0)
    outputs = torch.zeros(batch_size, MAX_LENGTH).to(device)

    src_embed = self.input_embedding(src).to(device) # (batch_size, seq_len, embed_dim)
    src_enc = self.encoder(src_embed).to(device) # (batch_size, seq_len, embed_dim)
    # src_padding_mask = self.generate_padding_mask(src).to(device) # (batch_size, seq_len)

    trg = torch.tensor([SOS_TOKEN_INDEX] * batch_size, device=device, dtype=torch.long).view(batch_size, 1)
    # trg.shape --> (batch_size, 1)

    for i in range(MAX_LENGTH):
      trg_mask = self.generate_square_subsequent_mask(i+1).to(device)
      src_mask = self.generate_src_mask(src.size(1), i+1).to(device)
      trg_enc = self.output_embedding(trg).to(device)
      output = self.decoder.forward(trg_enc, src_enc, trg_mask, src_mask)
      # output.shape --> (batch_size, seq_len, output_vocab_size)

      output = output[:, -1, :] # get the last element in the seq, shape --> (batch_size, output_vocab_size)
      pred_token = output.argmax(dim=-1).unsqueeze(1) # (batch_size, 1)
      trg = torch.cat([trg, pred_token], dim=1)
      outputs[:, i] = pred_token[:, 0]

    return outputs # shape --> (batch_size, max_length)

def inference(model, batch, output_vocab):

    # batch.shape --> (batch_size, seq_len)

    with torch.no_grad():
      output_tensor = model.forward_eval(batch.to(device)) # (output_tensor : batch_size, seq_len)
    
    output_list = []
    for i in range(output_tensor.size(0)):
        output_tokens = [english_output_vocab.index2word[idx] for idx in output_tensor[i].tolist()]
        if '<PAD>' in output_tokens:
            output_tokens = output_tokens[:output_tokens.index('<PAD>')]
        elif '<EOS>' in output_tokens:
            output_tokens = output_tokens[:output_tokens.index('<EOS>')]

        
        output_list.append(' '.join(output_tokens))

    return output_list

# loading the vocabs

model = Transformer(len(hindi_input_vocab), len(english_output_vocab), D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT)
model.load_state_dict(torch.load('model.params', map_location=device))
model = model.to(device)

model.eval()

ans_list = []

for batch in tqdm(test_loader):

  my_ans = inference(model, batch, english_output_vocab)
  ans_list += my_ans

with open('answer.txt', "a") as f:
  for my_ans in ans_list:
    f.write(f"{my_ans}\n")





