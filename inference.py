import pickle

import numpy as np
import torch

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertForSequenceClassification, AutoTokenizer


class EmotionRecognizer:

    def __init__(self, model_path, labels_mapping_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(labels_mapping_path, "rb") as f:
            mapper: MultiLabelBinarizer = pickle.load(f)
        
        self.type_map = {
            'input_ids': torch.long,
            'attention_mask': torch.float,
            'token_type_ids': torch.long,
            'labels': torch.float
        }
        self.le = mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, text, treshold=0.5) -> str:
        predictions = self.get_raw_prediction(text, treshold=treshold)
        predictions = self.le.inverse_transform(predictions)
        return ", ".join(predictions[0])

    def recognize_batch(self, batch, t=0.5) -> list:
        predictions = self.get_raw_predictions(batch, treshold=t)
        predictions = self.le.inverse_transform(predictions)
        return [", ".join(pred) for pred in predictions]

    def get_raw_prediction(self, text, treshold=0.5) -> np.ndarray:
        """
        Returns the normalized preditions of the model in binary format.
        Normalized preditions are obtained by applying a sigmoid function
        to the raw preditions and then applying a treshold.
        """
        encodings = self.tokenizer(text, add_special_tokens=True)
        encodings = {
            k: torch.tensor(v, dtype=self.type_map[k]) \
              .unsqueeze(0) \
              .to(self.device) for k, v in encodings.items()
        }

        with torch.no_grad():
            output = self.model(**encodings)
            predictions = torch.sigmoid(output.logits)
            predictions = torch.where(predictions > treshold, 1, 0)
            return predictions.detach().numpy()
    
    def get_raw_predictions(self, batch, treshold=0.5)-> np.ndarray:
      encodings = self.tokenizer(
          batch,
          padding="max_length",
          max_length=512,
          add_special_tokens=True
      )
      encodings = {
          k:torch.tensor(v, dtype=self.type_map[k]) \
            .to(self.device) for k, v in encodings.items()
      }
      with torch.no_grad():
        output = self.model(**encodings)
        predictions = torch.sigmoid(output.logits)
        predictions = torch.where(predictions > treshold, 1, 0)
        return predictions.detach().numpy()
