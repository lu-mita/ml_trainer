import re

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer
import json
import os
import spacy
import string


spacy_nlp = spacy.load('it_core_news_sm')
emoji_dict_path = os.path.join(os.path.dirname(__file__), 'emoji_dictionary.json')

with open(emoji_dict_path, 'r', encoding="utf-8") as fd:
    emoji_dict = json.load(fd)


EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


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



def preprocess_text(text: str, do_lower_case: bool = False, strategy: str = "transform") -> str:
    if strategy == "transform":
        text = str(" ".join(text_processor.pre_process_doc(text)))
        if do_lower_case: text = text.lower() 
        # text = re.sub(r'[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]', ' ', text)
        dictionary = emoji_dict
        for emoji in dictionary.keys():
            text = text.replace(emoji, f"<{dictionary[emoji]}>")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
        text = re.sub(r'^\s', '', text)
        text = re.sub(r'\s$', '', text)
        return text
    else:
        doc = spacy_nlp(text)
        out = ''
        for t in doc:
            bad = t.like_url
            bad |= t.text.startswith('@')
            bad |= t.text.startswith('#')
            bad |= t.like_email
            bad |= t.like_num
            if not bad:
                out += t.text_with_ws
        out = re.sub(r"""([?.!,;"'])""", r" ", out)
        out = re.sub(r'(\w)\1{2,}', r'\1\1', out)
        out = out.translate(str.maketrans('', '', string.punctuation))
        out = EMOJI_PATTERN.sub(r'', out)
        out = _RE_COMBINE_WHITESPACE.sub(" ", out).strip()
        return out


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
