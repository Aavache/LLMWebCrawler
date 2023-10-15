import torch


class BaseLanguageModel:
    def __init__(
        self,
        embedding_aggr_fn_name: str = "mean",
    ):
        self.embedding_aggr_fn = embedding_aggr_fn_name

    @property
    def max_token_length(self):
        raise NotImplementedError

    @property
    def embed_size(self):
        raise NotImplementedError

    def _chunk_tokens(self, tokens):
        chunks = []
        for i in range(0, len(tokens), self.max_token_length):
            chunks.append(tokens[i : i + self.max_token_length])
        return chunks

    def _aggregate_embeddings(self, embeddings, dim=0):
        if self.embedding_aggr_fn == "mean":
            embedding_aggr = embeddings.mean(dim=dim)
        elif self.embedding_aggr_fn == "max":
            embedding_aggr = embeddings.max(dim=dim)
        else:
            raise NotImplementedError("The embedding aggregation function `{self.embedding_aggr_fn}` is not allowed")
        return embedding_aggr

    def text_to_embedding(self, text):
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Preprocess text just in case the number of tokens is too large
        chunks = self._chunk_tokens(tokens)

        embedding_per_chunk = []
        for chunk in chunks:
            # Convert tokens to PyTorch tensors
            input_ids = torch.tensor(chunk).unsqueeze(0)  # Batch size of 1
            # Get BERT model embeddings
            with torch.no_grad():
                outputs = self.model(input_ids).last_hidden_state.squeeze(0)
                # Extract embeddings from the model output
                embeddings = self._aggregate_embeddings(outputs)
            # Convert embeddings to NumPy array
            embedding_per_chunk.append(embeddings)

        # Stacking the embeddings all from chunks
        embedding_stack = torch.stack(embedding_per_chunk)
        embedding_output = self._aggregate_embeddings(embedding_stack).squeeze(0)

        return embedding_output.numpy()
