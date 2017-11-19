import numpy as np
from ktree import ktree,KTreeOptions,utils

k = utils.load_ktree('save_ktree.pickle')
print k.model,k.root,k.depth,k.nearest_neighbor([57,8])
