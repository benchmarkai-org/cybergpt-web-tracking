class BaseDataset:
    def __init__(self):
        pass


class BaseNodeDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.vocab = set()

    def get_vocab_size(self) -> int:
        return len(self.vocab)
