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

    src_embed = self.input_embedding(src) # (batch_size, seq_len, embed_dim)
    src_enc = self.encoder(src_embed) # (batch_size, seq_len, embed_dim)

    src_padding_mask = self.generate_padding_mask(src).to(device) # (batch_size, seq_len)
    trg_padding_mask = self.generate_padding_mask(trg).to(device) # (batch_size, seq_len)

    trg_embed = self.output_embedding(trg) # (batch_size seq_len, embed_dim)
    trg_dec = self.decoder.forward(tgt=trg_embed, memory=src_embed, tgt_mask=trg_mask, memory_mask=src_mask, tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=src_padding_mask)

    # trg_dec.shape --> (batch_size, tgt_seq_len, embed_dim)

    output = self.out(trg_dec) # (batch_size, tgt_seq_len, output_vocab_size)
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

def train_batch(model, optimizer, criterion, input_tensors, target_tensors):

  optimizer.zero_grad()

  input_tensors = input_tensors.to(device) # shape : (batch_size, seq_len)
  target_tensors = target_tensors.to(device) # shape : (batch_size, seq_len)

  outputs = model(input_tensors, target_tensors[:, :-1]) 
  # outputs.shape --> (batch_size, seq_len-1, output_vocab_size)
  
  outputs = outputs.view(-1, outputs.size(-1))
  # shape : (batch_size * (seq_len-1), output_vocab_size)

  target_tensors = target_tensors[:, 1:].contiguous().view(-1)

  loss = criterion(outputs, target_tensors)

  loss.backward()

  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
  # clipping the grad to avoid exploding gradients problem

  optimizer.step()
  # update the params

  return loss.item()

class Mydat(Dataset):

  def __init__(self, input_seqs, output_seqs):
    super().__init__()
    self.input_seqs = input_seqs
    self.output_seqs = output_seqs

  def __len__(self):
    return len(self.input_seqs)

  def __getitem__(self, index):
    return self.input_seqs[index], self.output_seqs[index]

with open('hindi_input_vocab.pkl', 'rb') as f: 
    hindi_input_vocab = pickle.load(f)

with open('english_output_vocab.pkl', 'rb') as f: 
    english_output_vocab = pickle.load(f)

with open('dataset.pkl', 'rb') as f: 
    dataset = pickle.load(f)

# GLOABL AND HYPERPARAMS:

BATCH_SIZE = 32
MAX_LENGTH = 64
D_MODEL = 200
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
LR = 0.0005

if __name__ == "__main__":

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(len(train_loader))

    batch = next(iter(train_loader))
    print(type(batch))
    print(batch[0].shape) # input_tensors
    print(batch[1].shape) # output_tensors

    model = Transformer(len(hindi_input_vocab), len(english_output_vocab), D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss(ignore_index=english_output_vocab.word2index['<PAD>'])

    # training the model

    for epoch in tqdm(range(MAX_EPOCHS)):

        epoch_loss = 0

        for batch in tqdm(train_loader):

            input_tensors = batch[0].to(device)
            target_tensors = batch[1].to(device)

            batch_loss = train_batch(model, optimizer, criterion, input_tensors, target_tensors)

            epoch_loss += loss

        print(f"epoch = {epoch}/{max_epochs}, LOSS = {epoch_loss/len(dataloader)}")

        with open('training_loss1.txt', "a") as f: 
                my_dict = {"epoch": epoch, "max_epochs": max_epochs, "epoch_loss": epoch_loss/len(dataloader)}
                f.write(f"{my_dict}\n")

        torch.save(model.state_dict().cpu(), 'model.params')