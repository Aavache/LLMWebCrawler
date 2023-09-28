from transformers import BertTokenizer, BertModel


class BaseLanguageModel:
    def __init__(self, model_name: str, embedding_aggr_fn: str = "mean"):
        super().__init__(embedding_aggr_fn)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

