import torch 
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import sys 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_sentences = ['Hello, how are you?', 'What is your name?']
output_sentences = ['नमस्ते, आप कैसे हैं?', 'आपका नाम क्या है?']

# Tokenize input sentences and convert to BERT embeddings
input_tokens = tokenizer(input_sentences, padding='max_length', truncation=True, return_tensors='pt', max_length=256)
input_ids = input_tokens['input_ids']  # shape: (batch_size, seq_len)
input_mask = input_tokens['attention_mask']  # shape: (batch_size, seq_len)

with torch.no_grad():
    bert_outputs = bert_model(input_ids, input_mask)['last_hidden_state']  # shape: (batch_size, seq_len, bert_hidden_size)

class Seq2Seq(nn.Module):

    def __init__(self, input_dim, hidden_dim ,output_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = nn.GRU(input_dim, hidden_dim, n_layers, dropout=self.dropout)
        self.decoder = nn.GRU(input_dim, output_dim, n_layers, dropout=self.dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, output_seq, teacher_forcing_ratio=0.5):
        
        batch_size = input_seq.shape[1]
        max_len = output_seq.shape[0]

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(input_seq.device)

        enc_output, enc_hidden = self.encoder(input_seq, hidden)
        dec_input = torch.tensor([[0]] * batch_size).to(input_seq.device)

        dec_outputs = torch.zeros(max_len, batch_size, self.output_dim).to(input_seq.device)

        # loop through decoder sequence
        for t in range(1, max_len):
            
            dec_output, hidden = self.decoder(dec_input, hidden)

            # Predict next token
            dec_output = self.fc_out(dec_output)
            dec_output = self.softmax(dec_output)
            dec_outputs[t] = dec_output

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top1 = dec_output.max(1)[1]

            # Update decoder input with ground truth or predicted token
            dec_input = output_seq[t] if use_teacher_forcing else top1

        return dec_outputs

print(bert_outputs.shape)
# sys.exit()

INPUT_DIM = bert_outputs.shape[2]  # BERT hidden size
HIDDEN_DIM = 256
OUTPUT_DIM = len(tokenizer.vocab)
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 2

seq2seq_model = Seq2Seq(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=LEARNING_RATE)

# Train model
for epoch in range(N_EPOCHS):
    running_loss = 0.0
    seq2seq_model.train()
    for i in range(len(input_sentences)):
        input_seq = bert_outputs[i].unsqueeze(1)
        output_seq = torch.tensor(tokenizer.encode(output_sentences[i])).unsqueeze(1).to(device)

        optimizer.zero_grad()

        # Pass input and output sequences through model
        output = seq2seq_model(input_seq, output_seq)

        # Compute loss and perform backpropagation
        output_dim = seq2seq_model.fc_out.out_features
        loss = criterion(output[1:].view(-1, output_dim), output_seq[1:].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}: Loss = {running_loss / len(input_sentences):.4f}')