import os
import tempfile
import pickle
import re

class IpfsWeightsLoader():
  def __init__(self, ipfs_api='/ip4/127.0.0.1/tcp/5001'):
    self.direct = (ipfs_api == None)

    if not self.direct:
      self.ipfs_api = ipfs_api

  def load(self, id):
    with tempfile.TemporaryDirectory() as tempdir:
      weights_path = os.path.join(tempdir, 'weights.pkl')

      if self.direct:
        os.system(f"ipfs get -o {weights_path} {id}")
      else:
        os.system(f"ipfs get --api {self.ipfs_api} -o {weights_path} {id}")

      with open(weights_path, 'rb') as fp:
        weights = pickle.load(fp)
        return weights

  def store(self, weights):
    with tempfile.TemporaryDirectory() as tempdir:
      weights_path = os.path.join(tempdir, 'weights.pkl')
      with open(weights_path, 'wb') as fp:
        pickle.dump(weights, fp)

      out = None

      if self.direct:
        out = os.popen(f'ipfs add {weights_path}').read().strip().split('\n').pop()
      else:
        out = os.popen(f'ipfs add --api {self.ipfs_api} -q {weights_path}').read().strip().split('\n').pop()

      match = re.search(r'added\s+(\S+)', out)

      if match:
          out = match.group(1)
      
      return out

  def store_from_path(self, weights_path):
      if self.direct:
        out = os.popen(f'ipfs add {weights_path}').read().strip().split('\n').pop()
      else:
        out = os.popen(f'ipfs add --api {self.ipfs_api} -q {weights_path}').read().strip().split('\n').pop()

      return out

