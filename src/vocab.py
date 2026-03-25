class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def word2index(self, word, train=True):
        if word not in self.word2idx:
            if train:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
            else:
                return -1
        return self.word2idx[word]

    def to_dict(self):
        return {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word
        }

    @classmethod
    def from_dict(cls, vocab_dict):
        vocab = cls()
        vocab.word2idx = vocab_dict['word2idx']
        vocab.idx2word = vocab_dict['idx2word']
        return vocab

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.word2idx.get(idx, -1)
        elif isinstance(idx, int):
            if idx < len(self.idx2word):
                return self.idx2word[idx]
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("Expected string or integer index")
