import os
import torchaudio
from torch.utils.data import Dataset
import logging

class AudioDataset(Dataset):
    label_to_index = {'yes': 0, 'no': 1, 'silence': 2, 'unknown': 3}
    
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.labels = [self.get_label_from_path(fp) for fp in file_paths]
        logging.basicConfig(level=logging.INFO)
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            logging.error(f"Error loading audio file {file_path}: {e}")
            raise e
        return waveform.squeeze(), self.labels[idx]

    @staticmethod
    def get_label_from_path(file_path):
        if 'silence' in file_path:
            return AudioDataset.label_to_index['silence']
        else:
            label_name = os.path.basename(os.path.dirname(file_path))
            return AudioDataset.label_to_index.get(label_name, AudioDataset.label_to_index['unknown'])