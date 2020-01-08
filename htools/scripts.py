import sys
import importlib


def pimport(path, name, from_=None):
    multiple = isinstance(name, (list, tuple))
    
    if from_ is None:
        module = name
    else:
        module = from_
    
    from pathlib import Path
    
    p = Path(path)
    p = p.absolute()
    
    sys.path.append(p.as_posix())
    
    try:
        imported = importlib.import_module(module)
    finally:
        sys.path.pop(-1)
    
    if from_ and multiple:
        names = name
        importeds = [getattr(imported, name) for name in names]
        return importeds
    if from_ and not multiple:
        attribute = getattr(imported, name)
        return attribute
    return imported


def n_of(thing, n=1):
    for i in range(n):
        yield thing
    while True:
        yield None


def next_exp_name(prev_name):
    *bulk, end = prev_name
    num = ord(end) - 97
    if num >= 25:
        bulk.append(end)
    end = chr(((num + 1) % 26) + 97)
    bulk.append(end)
    return ''.join(bulk)


def num_to_alpha(num):
    num += -1
    letters = []
    while num >= 0:
        rem = (num % 26) + 1
        letters.append(chr(rem + 96))
        num -= (rem)
    return ''.join(reversed(letters))

def flatten(thing, unwrap=(list, tuple), _total=None):
    if _total is None:
        _total = []
    for item in thing:
        if isinstance(item, unwrap):
            flatten(item, unwrap=unwrap, _total=_total)
        else:
            _total.append(item)
    return _total