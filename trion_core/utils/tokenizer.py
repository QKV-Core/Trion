import torch

class SimpleCharTokenizer:
    def __init__(self, text=None):
        if text is None:
            # Varsayılan İngilizce karakter seti
            chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n"
        else:
            chars = sorted(list(set(text)))
            
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        print(f"🔤 Tokenizer Ready. Vocab Size: {self.vocab_size}")

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([self.itos[t] for t in tokens])