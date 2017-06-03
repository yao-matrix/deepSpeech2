from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

class arglist:
    cpu = 'knl'

def setenvs(inargv):
    args = arglist()
    for i in range(0, len(inargv) - 1):
        if inargv[i] == '--cpu' :
            args.cpu = inargv[i + 1]
    assert (args.cpu == 'knl' or args.cpu == 'bdw')
    # print 'Using cpu', args.cpu
    # print 'Groups set to', args.groups
    if (args.cpu == 'bdw'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    else:
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        os.environ["OMP_DYNAMIC"] = "false"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    return args
