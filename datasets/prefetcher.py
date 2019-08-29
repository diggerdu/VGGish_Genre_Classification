import torch
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            raise StopIteration()
            return
            
    def next(self):
        self.preload()
        return self.batch

    def __next__(self):
        return self.next()
    
    def __iter__(self):
        return self
