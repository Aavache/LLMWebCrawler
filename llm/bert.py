from transformers import BertModel, BertTokenizer

from llm.base import BaseLanguageModel

MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LEN = 512


class BaseLanguageModel(BaseLanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME)

    @property
    def max_token_length(self):
        return MAX_TOKEN_LEN
