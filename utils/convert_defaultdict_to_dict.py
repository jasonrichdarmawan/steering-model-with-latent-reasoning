def convert_defaultdict_to_dict(d):
  """
  Recursively convert defaultdicts to dicts
  """
  if isinstance(d, dict):
    d = {
      k: convert_defaultdict_to_dict(v)
      for k, v in d.items()
    }
  return d