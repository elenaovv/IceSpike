from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm

class TensorDataset(Dataset):
    def __init__(self, data: str):
        super(TensorDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        embedding = self.data[index][0]
        label = int(self.data[index][1])
        return embedding, label

class RateDataset(Dataset):
    def __init__(self, data: str):
        super(RateDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        rate_code = self.data[index][0]
        label = int(self.data[index][1])
        return rate_code, label

class TxtDataset(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        # print(temp)
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label

class TxtDataset2(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset2, self).__init__()
        with open(data_path, 'r', encoding='utf-8') as fin:
            self.lines = fin.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index].strip()
        # Split the line at the last tab
        parts = line.rsplit('\t', 1)  # rsplit splits from the right
        if len(parts) != 2:
            raise ValueError(f"Line {index} is malformed: {line}")
        sentence = parts[0]  # The text part
        label = int(parts[1])  # The label part, converted to integer
        return sentence, label

class TextDataset(Dataset):
    def __init__(self, raw_dataset):
        super(TextDataset, self).__init__()
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]["text"]

class IGCTextDataset(Dataset):
    def __init__(self, json_path):
        super(IGCTextDataset, self).__init__()
        self.json_path = json_path
        self.dataset_length = self._get_dataset_length()

    def _get_dataset_length(self):
        count = 0
        with open(self.json_path, 'r', encoding='utf-8') as file:
            for _ in file:
                count += 1
        return count

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index: int):
        # Read the file line by line until the desired index is reached
        with open(self.json_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == index:
                    json_object = json.loads(line)
                    return json_object["text"]
        # Just in case the index is out of bounds
        raise IndexError("Index out of bounds")
    
class ChnWikiDataset(Dataset):
    def __init__(self, data_path: str):
        super(ChnWikiDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        return line

class SentencePairDataset(Dataset):
    def __init__(self, data_path: str):
        super(SentencePairDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        # print(temp)
        sentence_pair = [temp[0], temp[1]]
        label = float(temp[2])
        return sentence_pair, label