import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import colors
from matplotlib import cm
from matplotlib import patches as mpatches
import eval_help as eh
import sys

if len(sys.argv) < 3:
    print("please add directory name and number of plots")
    sys.exit(1)

tag = str(sys.argv[1])
num = int(sys.argv[2])

eh.plot_density_profiles_2d_3d(tag,num)
