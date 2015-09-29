import numpy as np
from scipy import sparse as sp 

def _get_sparse_save_kwargs(A):
  return {
    "data": A.data,
    "indices": A.indices,
    "indptr": A.indptr,
    "shape": A.shape
  }

def save_csr_matrix(filepath, A):
  np.savez(filepath, **_get_sparse_save_kwargs(A))

def save_csc_matrix(filepath, A):
  np.savez(filepath, **_get_sparse_save_kwargs(A))

def _get_sparse_load_args(obj):
  return [(obj["data"], obj["indices"], obj["indptr"]), obj["shape"]]

def load_csr_matrix(filepath):
  obj = np.load(filepath)
  return sp.csr_matrix(*_get_sparse_load_args(obj))

def load_csc_matrix(filepath):
  obj = np.load(filepath)
  return sp.csc_matrix(*_get_sparse_load_args(obj))

class _SparseMatrixBuilder(object):
  def __init__(self):
    self.data = []
    self.indices = []
    self.indptr = [0]

  def _add_elements(self, data, indices):
    self.data += list(data)
    self.indices += list(indices)
    self.indptr.append(len(self.data))

class CSRMatrixBuilder(_SparseMatrixBuilder):
  def __init__(self):
    self.add_row = self._add_elements
    super(CSRMatrixBuilder, self).__init__()

  def get_matrix(self, shape=None, dtype=np.float32):
    if shape is None:
      shape = (len(self.indptr) - 1, max(self.indices) + 1)
    new_data = np.array(self.data, dtype=dtype)
    return sp.csr_matrix((new_data, self.indices, self.indptr), shape)

class CSCMatrixBuilder(_SparseMatrixBuilder):
  def __init__(self):
    self.add_column = self._add_elements
    super(CSCMatrixBuilder, self).__init__()

  def get_matrix(self, shape=None, dtype=np.float32):
    if shape is None:
      shape = (max(self.indices) + 1, len(self.indptr) - 1)
    new_data = np.array(self.data, dtype=dtype)
    return sp.csc_matrix((new_data, self.indices, self.indptr), shape)

