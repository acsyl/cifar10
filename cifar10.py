from __future__ import absolute_import
import numpy as np
import os
import sys
import six
from six.moves import cPickle
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
import hashlib
import zipfile
import tarfile
import shutil


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
  """Calculates a file sha256 or md5 hash.
  # Example
  ```python
      >>> from keras.data_utils import _hash_file
      >>> _hash_file('/path/to/file.zip')
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  ```
  # Arguments
      fpath: path to the file being validated
      algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
          The default 'auto' detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.
  # Returns
      The file hash
  """
  if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
    hasher = hashlib.sha256()
  else:
    hasher = hashlib.md5()

  with open(fpath, 'rb') as fpath_file:
    for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
      hasher.update(chunk)

  return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
  """Validates a file against a sha256 or md5 hash.
  # Arguments
      fpath: path to the file being validated
      file_hash:  The expected hash string of the file.
          The sha256 and md5 hash algorithms are both supported.
      algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
          The default 'auto' detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.
  # Returns
      Whether the file is valid
  """
  if ((algorithm is 'sha256') or
        (algorithm is 'auto' and len(file_hash) is 64)):
    hasher = 'sha256'
  else:
    hasher = 'md5'

  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
    return True
  else:
    return False


def _extract_archive(file_path, path='.', archive_format='auto'):
  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
  # Arguments
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
  # Returns
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format is 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, six.string_types):
    archive_format = [archive_format]

  for archive_type in archive_format:
    if archive_type is 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type is 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError,
                KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
  """Downloads a file from a URL if it not already in the cache.
  By default the file at the url `origin` is downloaded to the
  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
  and given the filename `fname`. The final location of a file
  `example.txt` would therefore be `~/.keras/datasets/example.txt`.
  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
  Passing a hash will verify the file after download. The command line
  programs `shasum` and `sha256sum` can compute the hash.
  # Arguments
      fname: Name of the file. If an absolute path `/path/to/file.txt` is
          specified the file will be saved at that location.
      origin: Original URL of the file.
      untar: Deprecated in favor of 'extract'.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of 'file_hash'.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are 'md5', 'sha256', and 'auto'.
          The default 'auto' detects the hash algorithm in use.
      extract: True tries extracting the file as an Archive, like tar or zip.
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).
  # Returns
      Path to the downloaded file
  """
  if cache_dir is None:
    cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
  if md5_hash is not None and file_hash is None:
    file_hash = md5_hash
    hash_algorithm = 'md5'
  datadir_base = os.path.expanduser(cache_dir)
  if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
  datadir = os.path.join(datadir_base, cache_subdir)
  if not os.path.exists(datadir):
    os.makedirs(datadir)

  if untar:
    untar_fpath = os.path.join(datadir, fname)
    fpath = untar_fpath + '.tar.gz'
  else:
    fpath = os.path.join(datadir, fname)

  download = False
  if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
    if file_hash is not None:
      if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
        print('A local file was found, but it seems to be '
              'incomplete or outdated because the ' + hash_algorithm +
              ' file hash does not match the original value of ' +
              file_hash + ' so we will re-download the data.')
        download = True
  else:
    download = True

  if download:
    print('Downloading data from', origin)

    class ProgressTracker(object):
      # Maintain progbar for the lifetime of download.
      # This design was chosen for Python 2.7 compatibility.
      progbar = None

    def dl_progress(count, block_size, total_size):
      if ProgressTracker.progbar is None:
        if total_size is -1:
          total_size = None
        pass
      else:
        pass

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
      except HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
    ProgressTracker.progbar = None

  if untar:
    if not os.path.exists(untar_fpath):
      _extract_archive(fpath, datadir, archive_format='tar')
    return untar_fpath

  if extract:
    _extract_archive(fpath, datadir, archive_format)

  return fpath


def load_batch(fpath, label_key='labels'):
  """Internal utility for parsing CIFAR data.
  # Arguments
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.
  # Returns
      A tuple `(data, labels)`.
  """
  f = open(fpath, 'rb')
  if sys.version_info < (3,):
    d = cPickle.load(f)
  else:
    d = cPickle.load(f, encoding='bytes')
    # decode utf8
    d_decoded = {}
    for k, v in d.items():
      d_decoded[k.decode('utf8')] = v
    d = d_decoded
  f.close()
  data = d['data']
  labels = d[label_key]

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels


def load_data():
  """Loads CIFAR10 dataset.
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  dirname = 'cifar-10-batches-py'
  origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  num_train_samples = 50000

  x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.zeros((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    data, labels = load_batch(fpath)
    x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
    y_train[(i - 1) * 10000: i * 10000] = labels

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  x_train = x_train.transpose(0, 2, 3, 1)
  x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)


class Cifar10(object):
  def __init__(self, batch_size=64, one_hot=False, test=False, shuffle=True):
    (x_train, y_train), (x_test, y_test) = load_data()
    if test:
      images = x_test
      labels = y_test
    else:
      images = x_train
      labels = y_train
    if one_hot:
      one_hot_labels = np.zeros((len(labels), 10))
      one_hot_labels[np.arange(len(labels)), labels.flatten()] = 1
      labels = one_hot_labels
    self.shuffle = shuffle
    self._images = images
    self.images = self._images
    self._labels = labels
    self.labels = self._labels
    self.batch_size = batch_size
    self.num_samples = len(self.images)
    if self.shuffle:
      self.shuffle_samples()
    self.next_batch_pointer = 0

  def shuffle_samples(self):
    image_indices = np.random.permutation(np.arange(self.num_samples))
    self.images = self._images[image_indices]
    self.labels = self._labels[image_indices]

  def get_next_batch(self):
    num_samples_left = self.num_samples - self.next_batch_pointer
    if num_samples_left >= self.batch_size:
      x_batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      y_batch = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      self.next_batch_pointer += self.batch_size
    else:
      x_partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
      y_partial_batch_1 = self.labels[self.next_batch_pointer:self.num_samples]
      if self.shuffle:
        self.shuffle_samples()
      x_partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
      y_partial_batch_2 = self.labels[0:self.batch_size - num_samples_left]
      x_batch = np.vstack((x_partial_batch_1, x_partial_batch_2))
      y_batch = np.vstack((y_partial_batch_1, y_partial_batch_2))
      self.next_batch_pointer = self.batch_size - num_samples_left
    return x_batch, y_batch


cifar10_train = Cifar10(batch_size=100, one_hot=True, test=False,
                        shuffle=True)  # Load Cifar10 train set with batch size 100 and one hot label.
cifar10_test = Cifar10(batch_size=100, one_hot=False, test=True, shuffle=False)  # Load Cifar10 test set
cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels  # Get all images and labels of the test set.
batch_x, batch_y = cifar10_train.get_next_batch()  # get the next batch of Cifar10 train set
