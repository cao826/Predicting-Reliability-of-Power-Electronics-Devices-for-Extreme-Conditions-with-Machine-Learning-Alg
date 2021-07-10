import re
import glob
import pandas as pd
import numpy as np
import math
from decimal import *
import operator
import matplotlib.pyplot as plt
import seaborn as sn
import module as mod

#df_multi_model_pf_preds, df_multi_model_pf_preds_acc =  mod.pf_prediction_generator(mod.df_pre_static_all, 'OS', False, True)

#mod.manufacturer_dendrogram('full_name', True)

#pca = mod.pca_grapher(mod.df_pre_static_all, 'DS', 'MN', True, 'ALL', True, 'eh-small')

# By Manufacturer where entire manufacturers are used as test group (results skewed)
s_mtpf_lr = mod.predict_manutest_pf('LR')
s_mtpf_rfc = mod.predict_manutest_pf('RFC')
s_mtpf_gbc = mod.predict_manutest_pf('GBC')

# # Concatenates them
bp_manutest_concat = pd.concat([s_mtpf_lr, s_mtpf_gbc, s_mtpf_rfc])
bp_manutest_concat = bp_manutest_concat.reset_index()
bp_manutest_concat = bp_manutest_concat.rename({'index': 'Manufacturer'}, axis='columns')
mod.boxplotter(bp_manutest_concat, 'Manufacturer', 'Accuracy', 'Manufacturer', 'Accuracy',
            'Box Plot of Accuracies by Manufacturer', 'NewBoxPlot', True)
