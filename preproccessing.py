import re

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer
import json
import os
import string


with open(os.path.join(os.path.dirname(__file__), 'emoji_dictionary.json'), 'r', encoding="utf-8") as fd:
    emoji_translation = json.load(fd)


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'user', 'percent', 'money', 'phone', 'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag"},
    fix_html=True,  # fix HTML tokens
    segmenter="twitter",  # corpus from which the word statistics are going to be used  for word segmentation 
    corrector="twitter",  # corpus from which the word statistics are going to be used for spell correction
    unpack_hashtags=True,  # perform word segmentation on hashtags
    spell_correct_elong=True,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer().tokenize,
    dicts=[emoticons]
)


def preprocess_text(text: str, do_lower_case: bool = False) -> str:
    text = str(" ".join(text_processor.pre_process_doc(text)))
    if do_lower_case: text = text.lower() 
    # text = re.sub(r'[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]', ' ', text)
    for emoji in emoji_translation.keys():
        text = text.replace(emoji, f"<{emoji_translation[emoji]}>")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'^\s', '', text)
    text = re.sub(r'\s$', '', text)
    return text


def encode_data(model_name_or_path, text_samples: list, label_samples: list, max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    encodings = tokenizer(
        text_samples,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        max_length=max_length
    )
    num_samples = len(text_samples)
    encoded_dataset = {**encodings, 'labels': label_samples}
    output = []
    for i in range(num_samples):
        output.append(
            { key: encoded_dataset[key][i] for key in encoded_dataset.keys() }
        )
    return output
