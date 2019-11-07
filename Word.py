class Word():
    def __init__(self, name, context, tok_vec, label, cluster=None):
        self.name = name
        self.context = context
        self.tok_vec = tok_vec
        self.label = label
        self.cluster = cluster
