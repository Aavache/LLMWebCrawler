import torch


class BaseLanguageModel:
    def __init__(self, embedding_aggr_fn: str):
        self.embedding_aggr_fn = embedding_aggr_fn

    def _aggregate_embeddings(self, embeddings):
        if self.embedding_aggr_fn == "mean":
            embedding_aggr = embeddings.last_hidden_state.squeeze(0).mean(dim=0)
        elif self.embedding_aggr_fn == "max":
            embedding_aggr = embeddings.last_hidden_state.squeeze(0).max(dim=0)
        else:
            raise ValueError("The embedding aggregation function `{self.embedding_aggr_fn}` is not allowed")
        return embedding_aggr

    def text_to_embedding(self, text):
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        # Convert tokens to PyTorch tensors
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size of 1
        # Get BERT model embeddings
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Extract embeddings from the model output
            embeddings = self._aggregate_embeddings(outputs)
        # Convert embeddings to NumPy array
        embeddings_np = embeddings.numpy()
        return embeddings_np

