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
        self.le = mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, text, treshold=0.5) -> str:
        predictions = self.get_raw_predictions(text, treshold)
        predictions = self.le.inverse_transform(predictions)
        prediction = list(predictions[0])
        texual_predictions = ", ".join(prediction)
        return texual_predictions

    def get_raw_predictions(self, text, treshold=0.5) -> np.ndarray:
        """
        Returns the normalized preditions of the model in binary format
        Normalized preditions are obtained by applying a sigmoid function to the raw preditions and then applying a treshold
        """
        encodings = self.tokenizer(text, add_special_tokens=True)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0).long().to(self.device)
        attention_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0).long().to(self.device)
        token_type_ids = torch.tensor(encodings['token_type_ids']).unsqueeze(0).long().to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = torch.sigmoid(output.logits)
            predictions = torch.where(predictions > treshold, 1, 0)
            return predictions.detach().numpy()
