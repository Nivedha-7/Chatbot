class TempStore:
    def __init__(self):
        self.docs = []  # list of (text, vector)
 
    def add(self, texts, vectors):
        for t, v in zip(texts, vectors):
            self.docs.append((t, v))
 
    def clear(self):
        self.docs = []