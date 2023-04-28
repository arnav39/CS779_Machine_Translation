import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import torch.nn.functional as F
import spacy
from tqdm import tqdm
import sys
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
import pickle

class EncoderGRU(nn.Module):

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def forward(self, input, hidden):

    # input.shape : (1, )

    embedded = self.embedding(input).view(1, 1, -1)
    # embedded.shape = (1, 1, hidden_size)

    output, hidden = self.gru(embedded, hidden)
    # output.shape = (1, 1, hidden_size) , (seq_len, batch_size, hidden_size)
    # hidde.shape = (1, 1, hidden_size), (num_layers, batch_size, hidden_size)

    return output, hidden 

  def init_hidden(self):
    hidden = torch.zeros(1, 1, self.hidden_size) # (num_layers, batch_size, hidden_dim)
    return hidden
  
class DecoderGRU(nn.Module):

  def __init__(self, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    output = F.relu(output)
    output, hidden = self.gru(output, hidden)
    output = self.out(output[0])
    return output, hidden

  def init_hidden(self):
    hidden = torch.zeros(1, 1, self.hidden_size) # (num_layers, batch_size, hidden_size)
    return hidden

def train(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: EncoderGRU, decoder: DecoderGRU, encoder_optimizer, decoder_optimizer, criterion):

  input_tensor = input_tensor.to(device)
  target_tensor = target_tensor.to(device)
  
  encoder_hidden = encoder.init_hidden().to(device)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_tensor.size(0)
  target_length = target_tensor.size(0)

  encoder_outputs = torch.zeros(input_length, encoder.hidden_size).to(device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0, 0] 

  decoder_input = torch.LongTensor([0]).to(device)
  decoder_hidden = encoder_hidden

  for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    # print(decoder_output)
    # print(target_tensor[di])
    # sys.exit()
    single_loss = criterion(decoder_output, target_tensor[di].view(1))
    # print(single_loss)
    # sys.exit()
    loss += single_loss
    decoder_input = target_tensor[di]

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  ans = loss.detach().cpu().item()/target_length
  return ans

class Lang():

  def __init__(self, name, spacy_tokenizer):
    self.name = name
    self.word2index = {"<SOS>":0, '<EOS>': 1, "<UNK>": 2}
    self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>"}
    self.word2count = {}
    self.n_words = 3
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
    self.word2index = {"<SOS>":0, '<EOS>': 1, "<UNK>": 2}
    self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>"}
    self.word2count = {}
    self.n_words = 3
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
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

hidden_size = 100
learning_rate = 0.01
max_epochs = 10

if __name__ == "__main__":

    with open('inp_sent.pkl', 'rb') as f: 
        output_sent_list = pickle.load(f)

    with open('out_sent.pkl', 'rb') as f: 
        input_sent_list = pickle.load(f)

    print(type(input_sent_list))
    print(len(input_sent_list))

    print(type(output_sent_list))
    print(len(output_sent_list))

    with open('hindi_input_vocab.pkl', 'rb') as f:
        hindi_input_vocab = pickle.load(f)

    with open('english_output_vocab.pkl', 'rb') as f: 
        english_output_vocab = pickle.load(f)

    training_data = []

    for i in range(len(input_sent_list)):
        pair = (input_sent_list[i], output_sent_list[i])
        training_data.append(pair)

    print(len(training_data))
    print(training_data[0])


    encoder = EncoderGRU(len(hindi_input_vocab), hidden_size).to(device)
    decoder = DecoderGRU(hidden_size, len(english_output_vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    for epoch in tqdm(range(max_epochs)):

        epoch_loss = 0

        for pair in tqdm(training_data):

            input_tokens = hindi_input_vocab.tokenize_sentence(pair[0])
            target_tokens = english_output_vocab.tokenize_sentence(pair[1])

            input_tensor = torch.tensor([hindi_input_vocab.word2index[my_token] for my_token in input_tokens]).to(device)
            target_tensor = torch.tensor([english_output_vocab.word2index[my_token] for my_token in target_tokens]).to(device)

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            epoch_loss += loss

        print(f"epoch = {epoch}/{max_epochs}, LOSS = {epoch_loss/len(training_data)}")

        torch.save(encoder.state_dict(), 'encoder.params')
        torch.save(decoder.state_dict(), 'decoder.params')
