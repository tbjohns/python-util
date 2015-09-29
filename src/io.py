import cPickle

def pickle(obj, path):
  cPickle.dump(obj, open(path, "w"))

def unpickle(path):
  return cPickle.load(open(path))

