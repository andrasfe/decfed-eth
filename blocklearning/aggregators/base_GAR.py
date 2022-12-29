class _GAR:
  """ Base gradient aggregation rule class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      nbworkers Total number of workers
      nbbyzwrks Declared number of Byzantine workers
      args      Command line argument list
    """
    raise NotImplementedError

  def aggregate(self, gradients):
    """ Build the gradient aggregation operation of the given gradients.
    Args:
      gradients Computed gradient tensors
    Returns:
      Aggregated gradient tensor
    """
    raise NotImplementedError