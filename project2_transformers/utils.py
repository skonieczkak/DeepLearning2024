from torch.nn.utils.rnn import pad_sequence
import torch

def load_file_list(file_name):
    with open(file_name, 'r') as file:
        file_list = file.read().splitlines()
    return file_list


def collate_fn(batch):
    audios, labels = zip(*batch)
    audios_padded = pad_sequence([audio for audio in audios], batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)
    return audios_padded, labels