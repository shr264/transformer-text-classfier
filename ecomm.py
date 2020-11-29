"""
transformer text classifier

conda activate rasy-pDL
nohup python ecomm.py -m baseline_word_split_replace_nums -d /data/work/may28/ --device 0 --word_split True --replace_nums True> baseline_word_split_replace_nums.log &

author: Syed Rahman
"""

import numpy as np
import torch
from text_cleaner import normalize
from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data import Field, TabularDataset, BucketIterator, Dataset, Example
import torchtext.datasets as datasets
import torchtext.data as data
import torchtext
import torch.nn.functional as f
from torch import nn
import copy
import io
import os
import sys
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, help="model path")
parser.add_argument("-d", "--data_dir", type=str, help="data path")
parser.add_argument("--train", type=str,
                    help="train file name", default='train.txt')
parser.add_argument("--test", type=str,
                    help="test file name", default='test.txt')
parser.add_argument("--validate", type=str,
                    help="validate file path", default='validate.txt')
parser.add_argument("--device", type=str, help="cuda device number", default=0)
parser.add_argument("--num_words", type=int,
                    help="cuda device number", default=None)
parser.add_argument("--batch_size", type=int,
                    help="cuda device number", default=1024)
parser.add_argument("--dropout", type=float,
                    help="cuda device number", default=0.1)
parser.add_argument("--lr", type=float,
                    help="cuda device number", default=0.001)
parser.add_argument("--word_split", type=bool,
                    help="use ninja word splitter", default=False)
parser.add_argument("--replace_nums", type=bool,
                    help="replace numbers with words", default=False)
args = parser.parse_args()
print(args)


print(torch.__version__)
torch.cuda.is_available()


data_dir = args.data_dir
dscb_train_fn = args.train
dscb_validate_fn = args.validate
dscb_test_fn = args.test

device = torch.device(
    "cuda:"+args.device if torch.cuda.is_available() else "cpu")


def charNGramtokenizer(sentence, n=3):
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def wordTokenizer(sentence):
    return sentence.split()


def normalizedWordTokenizer(sentence, word_split, replace_nums):
    sentence = normalize(sentence, args.word_split, args.replace_nums)
    return sentence.split()


def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


dscb_train = pd.read_csv(os.path.join(
    data_dir, dscb_test_fn), sep='\x01', dtype=str)

max_len = 100

text = Field(sequential=True, tokenize=normalizedWordTokenizer,
             fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
dscb = Field(sequential=False, dtype=torch.long)
dsc = Field(sequential=False, dtype=torch.long)
b = Field(sequential=False, dtype=torch.long)

fields = []
for col in dscb_train.columns:
    if col == "cleaned_description":
        fields.append(("text", text))
    elif col == "dscb":
        fields.append(("dscb", dscb))
    elif col == "dsc":
        fields.append(("dsc", dsc))
    elif col == "omni_brand_id":
        fields.append(("b", b))
    else:
        fields.append((col, None))


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.
        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
        """
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV,
            '\x01': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(
                    f, delimiter='\t', **csv_reader_params)
            elif format == '\x01':
                reader = unicode_csv_reader(
                    f, delimiter='\x01', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv', '\x01'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(
                    make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


ds_train, ds_valid, ds_test = TabularDataset.splits(
    path=args.data_dir,
    train=args.train,
    validation=args.validate,
    test=args.test,
    format='\x01',
    skip_header=True,
    fields=fields)

print('train : ', len(ds_train))
print('test : ', len(ds_test))
print('train.fields :', ds_train.fields)

if args.num_words is None:
    num_words = None
else:
    num_words = args.num_words

text.build_vocab(ds_train, max_size=num_words, specials=['<pad>', '<unk>'])
dscb.build_vocab(ds_train)
dsc.build_vocab(ds_train)
b.build_vocab(ds_train)
vocab = text.vocab

print(len(dscb.vocab.itos))

if not os.path.exists('vocab'):
    os.makedirs('vocab')

if not os.path.exists('models'):
    os.makedirs('models')

save_vocab(text.vocab, 'vocab/cleaned_description_'+args.model_name+'.pkl')
save_vocab(dscb.vocab, 'vocab/dscb_cleaned_desc_'+args.model_name+'.pkl')
save_vocab(dsc.vocab, 'vocab/dsc_cleaned_desc_'+args.model_name+'.pkl')
save_vocab(b.vocab, 'vocab/b_cleaned_desc_'+args.model_name+'.pkl')

if args.batch_size is None:
    batch_size = 1024
else:
    batch_size = args.batch_size

train_loader, valid_loader, test_loader = BucketIterator.splits(
    (ds_train, ds_valid, ds_test), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False)

nn_Softargmax = nn.Softmax  # fix wrong name


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input

        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. They will be split into number of heads
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesnt saturate
        # (bs, n_heads, q_length, dim_per_head)
        Q = Q / np.sqrt(self.d_k)
        # (bs, n_heads, q_length, k_length)
        scores = torch.matmul(Q, K.transpose(2, 3))

        A = nn_Softargmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)

        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X num_heads X d_k)
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads
        # (bs, n_heads, q_length, dim_per_head)
        Q = self.split_heads(self.W_q(X_q), batch_size)
        # (bs, n_heads, k_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)
        # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)

        # Final linear layer
        H = self.W_h(H_cat)          # (bs, q_length, dim)

        return H, A


class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.dropout1 = nn.Dropout(p=p)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.k1convL2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, p)
        self.cnn = CNN(d_model, conv_hidden_dim, p)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):

        # Multi-head attention
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x)

        # Layer norm after adding the residual connection
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)

        # Second layer norm after adding residual connection
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + cnn_output)

        return out2


def create_sinusoidal_embeddings(nb_p, dim, E):
    theta = np.array([
        [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for p in range(nb_p)
    ])
    E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
    E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
    E.detach_()
    E.requires_grad = False
    E = E.to(device)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, max_position_embeddings, p):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, d_model)
        create_sinusoidal_embeddings(
            nb_p=max_position_embeddings,
            dim=d_model,
            E=self.position_embeddings.weight
        )

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            input_ids)                      # (bs, max_seq_length)

        # Get word embeddings for each input id
        word_embeddings = self.word_embeddings(
            input_ids)                   # (bs, max_seq_length, dim)

        # Get position embeddings for each position id
        position_embeddings = self.position_embeddings(
            position_ids)        # (bs, max_seq_length, dim)

        # Add them both
        embeddings = word_embeddings + \
            position_embeddings  # (bs, max_seq_length, dim)

        # Layer norm
        # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, input_vocab_size,
                 maximum_position_encoding, p=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embeddings(
            d_model, input_vocab_size, maximum_position_encoding, p)

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(
                d_model, num_heads, ff_hidden_dim, p))

    def forward(self, x):
        # Transform to (batch_size, input_seq_length, d_model)
        x = self.embedding(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)


class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size, num_answers):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size,
                               maximum_position_encoding=10000)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(d_model, num_answers)

    def forward(self, x):
        x = self.encoder(x)

        x, _ = torch.max(x, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        return x


class TransformerClassifierDSCB(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size, num_dsc, num_b, p):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size,
                               maximum_position_encoding=10000)
        self.dropout = nn.Dropout(p=p)
        self.dense = nn.Linear(d_model, num_dsc)
        self.dense1 = nn.Linear(d_model, num_b)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dropout(x)
        x0 = self.dense(x)
        x1 = self.dense1(x)
        return x0, x1


class TransformerClassifierHrclDSCB(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size, num_d, num_s, num_c, num_b, p):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size,
                               maximum_position_encoding=10000)
        self.dropout = nn.Dropout(p=p)
        self.dense0 = nn.Linear(d_model, num_d)
        self.dense1 = nn.Linear(d_model+num_d, num_s)
        self.dense2 = nn.Linear(d_model+num_d+num_s, num_c)
        self.dense3 = nn.Linear(d_model+num_d+num_s+num_c, num_b)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dropout(x)

        x_d = self.dense0(x)
        x_s = self.dense1(torch.cat((x, x_d), 1))
        x_c = self.dense2(torch.cat((x, x_d, x_s), 1))
        x_b = self.dense3(torch.cat((x, x_d, x_s, x_c), 1))
        return x_d, x_s, x_c, x_b


num_dsc = len(dsc.vocab.itos)
num_b = len(b.vocab.itos)
num_words = len(text.vocab.itos)

model = TransformerClassifierDSCB(num_layers=2, d_model=128, num_heads=8,
                                  conv_hidden_dim=128, input_vocab_size=num_words+2,
                                  num_dsc=num_dsc, num_b=num_b, p=args.dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
epochs = 100
t_total = len(train_loader) * epochs

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=3,
                                                       verbose=True)


def trainDSCB(train_loader, valid_loader):

    best_loss = 10**100

    for epoch in range(epochs):
        train_iterator, valid_iterator = iter(train_loader), iter(valid_loader)
        train_acc_dsc = 0
        train_acc_b = 0
        train_acc = 0
        model.train()
        losses = 0.0

        for idx, batch in enumerate(train_iterator):
            x = batch.text.to(device)
            y_dsc = batch.dsc.to(device)
            y_b = batch.b.to(device)

            out = model(x)  # ①

            loss = f.cross_entropy(out[0], y_dsc) + \
                f.cross_entropy(out[1], y_b)  # ②

            model.zero_grad()  # ③

            loss.backward()  # ④
            losses += loss.item()

            optimizer.step()  # ⑤

            train_acc_dsc += (out[0].argmax(1) == y_dsc).cpu().numpy().mean()
            train_acc_b += (out[1].argmax(1) == y_b).cpu().numpy().mean()
            train_acc += ((out[0].argmax(1) == y_dsc) &
                          (out[1].argmax(1) == y_b)).cpu().numpy().mean()

        print(f"Training loss at epoch {epoch} is {losses/idx}")
        print(f"Training accuracy for DSC: {train_acc_dsc/idx}")
        print(f"Training accuracy for B: {train_acc_b/idx}")
        print(f"Training accuracy: {train_acc/idx}")
        print('Evaluating on validation:')

        model.eval()
        acc_dsc = 0
        acc_b = 0
        acc = 0
        val_losses = 0.0
        for idx, batch in enumerate(valid_iterator):
            x = batch.text.to(device)
            y_dsc = batch.dsc.to(device)
            y_b = batch.b.to(device)

            out = model(x)
            loss = f.cross_entropy(out[0], y_dsc) + \
                f.cross_entropy(out[1], y_b)
            val_losses += loss.item()
            acc_dsc += (out[0].argmax(1) == y_dsc).cpu().numpy().mean()
            acc_b += (out[1].argmax(1) == y_b).cpu().numpy().mean()
            acc += ((out[0].argmax(1) == y_dsc) &
                    (out[1].argmax(1) == y_b)).cpu().numpy().mean()

        print(f"Validation accuracy for DSC: {acc_dsc/idx}")
        print(f"Validation accuracy for B: {acc_b/idx}")
        print(f"Validation accuracy: {acc/idx}")

        if val_losses < best_loss:
            print('Updating best model')
            best_loss = copy.deepcopy(val_losses)
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(),
                       'models/'+args.model_name)

        scheduler.step(loss.item())

    return best_model


def evaluateDSCB(data_loader, best_model):
    data_iterator = iter(data_loader)

    best_model.eval()
    acc_dsc = 0
    acc_b = 0
    acc = 0

    for idx, batch in enumerate(data_iterator):
        x = batch.text.to(device)
        y_dsc = batch.dsc.to(device)
        y_b = batch.b.to(device)

        out = model(x)
        loss = f.cross_entropy(out[0], y_dsc) + f.cross_entropy(out[1], y_b)

        acc_dsc += (out[0].argmax(1) == y_dsc).cpu().numpy().mean()
        acc_b += (out[1].argmax(1) == y_b).cpu().numpy().mean()
        acc += ((out[0].argmax(1) == y_dsc) &
                (out[1].argmax(1) == y_b)).cpu().numpy().mean()

    print(f"Eval accuracy for DSC: {acc_dsc/idx}")
    print(f"Eval accuracy for B: {acc_b/idx}")
    print(f"Eval accuracy: {acc/idx}")


if __name__ == '__main__':
    best_model = trainDSCB(train_loader, valid_loader)
    print('Evaluating on test:')
    evaluateDSCB(test_loader, best_model)
