## condact synthESizer on bigboy (was originally keras2-tf27 plus additional packages)
## condact keras2-tf24 on lane????

bigboy_or_lane = 'local'

if bigboy_or_lane == 'bigboy':
    dh_prefix = '/lane/dorsalhorn/'
    home_prefix = '/lane/home/'
elif bigboy_or_lane == 'local':
    # Use current directory structure
    dh_prefix = './data/'
    home_prefix = './'
else:
    dh_prefix = '/projects/pfenninggroup/singleCell/Macaque_SealDorsalHorn_snATAC-seq/'
    home_prefix = '/home/mleone2/'

datapath = dh_prefix + 'data/tidy_data/synthetic_design/testing'
figurepath = dh_prefix + 'figures/exploratory/synthetic_design/testing'

import sys
# Import from local helpers module instead of external synthESizer
import helpers as seq_functions
import helpers as saturation_mutagenesis_functions

#####

import dill

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

from Bio import SeqIO
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr

import pandas as pd
import numpy as np
import argparse
import math
import os
import pickle
import time
import importlib

import pyranges as pr
from scipy.stats import gmean

## newer cnn pipeline for bayesian
import sys
repo = home_prefix + "repos/cnn_pipeline"
# External cnn_pipeline path removed - using local models
from models import predict_with_uncertainty

############## set mutation and dropout inference parameters
satmut_iterations = 20 # number of saturation mutagenesis mutations per sequence
num_seed_seqs = 200 # number of seed sequences for adalead
adalead_runs = 50 # 20 # number of adalead initalizations (new combinations of seed sequences each time)
adalead_recombine_turns = 4 # 5 # number of adalead iterations per adalead initialization
num_dropout_trials=64

target = 'GLUT7'
avoid =  'EXC'
flip_positives = False

load_previous_session = True
if load_previous_session:
    session_dir = dh_prefix + 'data/tidy_data/synthetic_design/sessions/'
    dill.load_session(session_dir + target + 'vs' + avoid + '_satmut_adalead_session.pkl')

# TEMPORARY
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# Check if TensorFlow is using the GPU
if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA support.")
else:
    print("TensorFlow is not built with CUDA support.")


# GLUT5 fold 1: run-20250628_140332-ue5pfxms
# GLUT6 fold 1: run-20250626_074830-easg81ln
models_base_path = dh_prefix + 'data/tidy_data/celltype_specific_enhancers/models_cnn_forSynthetic/'


## grab opt models -- GLUT6vsGLUT5 snail models -- macaque and mouse
# folds 1 - 5 macaque
if target == 'GLUT6' and avoid == 'GLUT5':
    macaque_runs = ['run-20250710_141023-j0sfbkho', 'run-20250711_131015-xb9ik0tb', 'run-20250711_131025-3d4od5dt', 
                'run-20250711_131926-rgt0814r', 'run-20250711_131915-0g5duvum']
    mouse_runs = ['run-20250711_101311-tl8lixob', 'run-20250711_103914-tvlhrxrd', 'run-20250711_103918-thqpt82g', 
              'run-20250711_130129-zypo81lt', 'run-20250711_130136-8twa85ft']
if target == 'GLUT7' and avoid == 'EXC':
    macaque_runs = ['run-20250713_045255-xtwox1mx', 'run-20250713_134921-uw8r6ntr', 'run-20250713_140203-i88rki4x',
                    'run-20250713_141528-tbipstwq', 'run-20250713_151038-gqvx1gkq']
    mouse_runs = ['run-20250713_050604-djldf0aj', 'run-20250713_153202-ra80b5i4', 'run-20250713_160431-uz2hprpw',
                  'run-20250713_164217-9bogo9gg', 'run-20250713_165755-qarumm9c']

combined_runs = macaque_runs + mouse_runs


opt_models_paths = [models_base_path  + run + '/files/model-best.h5' for run in combined_runs]
opt_models_names = ['macaque_' + target + 'vs' + avoid + '_fold' + str(ii+1) for ii in range(5)] + \
                    ['mouse_' + target + 'vs' + avoid + '_fold' + str(ii+1) for ii in range(5)]

print('loading models')
opt_models = {}
for ii in range(len(opt_models_names)):
    opt_models_name = opt_models_names[ii]
    opt_models[opt_models_name] = load_model(opt_models_paths[ii], compile=False)

### validation models
model_df_path = dh_prefix + 'data/tidy_data/celltype_specific_enhancers/tables/xspecies_all_models_record.csv'
model_df = pd.read_csv(model_df_path)
model_df = model_df.set_index('model')
val_models_base_path = dh_prefix + 'data/tidy_data/celltype_specific_enhancers/models/'
if target == 'GLUT6' and avoid == 'GLUT5':
    val_model0 = val_models_base_path + model_df.at['GLUT6vsEXC', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model1 = val_models_base_path + model_df.at['GLUT6vsINH', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model2 = val_models_base_path + model_df.at['GLUT6vsmidVen', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model3 = val_models_base_path + model_df.at['GLUT6vsGLIA', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model4 = val_models_base_path + model_df.at['GLUT6vsGLUT5', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_models_paths = [val_model0, val_model1, val_model2, val_model3, val_model4]
    val_models_names = ['GLUT6vsEXC_classification_v1_fold1',
                        'GLUT6vsINH_classification_v1_fold1', 'GLUT6vsmidVen_classification_v1_fold1',
                        'GLUT6vsGLIA_classification_v1_fold1','GLUT6vsGLUT5_classification_v1_fold1']
if target == 'GLUT7' and avoid == 'EXC':
    val_model0 = val_models_base_path + model_df.at[target + 'vsEXC', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model1 = val_models_base_path + model_df.at[target + 'vsINH', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model2 = val_models_base_path + model_df.at[target + 'vsmidVen', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_model3 = val_models_base_path + model_df.at[target + 'vsGLIA', 'classification_v1_fold1_run_path'] + '/files/model-best.h5'
    val_models_paths = [val_model0, val_model1, val_model2, val_model3]
    val_models_names = ['GLUT7vsEXC_classification_v1_fold1',
                        'GLUT7vsINH_classification_v1_fold1', 'GLUT7vsmidVen_classification_v1_fold1',
                        'GLUT7vsGLIA_classification_v1_fold1']
val_models = {}
for ii in range(len(val_models_names)):
    val_models_name = val_models_names[ii]
    val_models[val_models_name] = load_model(val_models_paths[ii], compile=False)

### testSet models
# cross- species one model
if target == 'GLUT7' and avoid == 'EXC':
    runs = ['run-20250717_101626-gdhctzx6']
    tmp_avoid = 'EXC'
if target == 'GLUT6' and avoid == 'GLUT5':
    runs = ['run-20250717_095435-6jh5fy4p']
    tmp_avoid = 'EXC'

testSet_models_paths = [models_base_path  + run + '/files/model-best.h5' for run in runs]
testSet_models_names = ['testSet_crossSpecies_' + target + 'vs' + tmp_avoid + '_classification']

print('loading models')
testSet_models = {}
for ii in range(len(testSet_models_names)):
    testSet_models_name = testSet_models_names[ii]
    testSet_models[testSet_models_name] = load_model(testSet_models_paths[ii], compile=False)
testSet_model = None
if len(testSet_models_names) == 1:
    testSet_model = next(iter(testSet_models.values()))

##fasta
fasta_path = dh_prefix + 'data/tidy_data/celltype_specific_enhancers/fasta/'
fasta_diff_name = 'macaque_dh_celltypes_split_positives/rheMac10_' + target + 'vs' + avoid + '_positive/fold1/train.fa'
fasta_diff_negatives = 'macaque_dh_celltypes_split_negatives/rheMac10_' + target + 'vs' + avoid + '_negative/fold1/train.fa'

if target == 'GLUT6' and avoid == 'GLUT5':
    fasta_skor2_103 = fasta_path + 'candidates/SKOR2.103.original_previousNameGLUT6.97.fa'
    skor2_103_seq = seq_functions.get_fasta_seqs(fasta_skor2_103)
    skor2_103_seq = skor2_103_seq[:,:,:,None]
    fasta_skor2_55 = fasta_path + 'candidates/SKOR2.55.fa'
    skor2_55_seq = seq_functions.get_fasta_seqs(fasta_skor2_55)
    skor2_55_seq = skor2_55_seq[:,:,:,None]

seed_fasta = fasta_path + fasta_diff_name
seed_pool_seqs = seq_functions.get_fasta_seqs(seed_fasta)
seed_pool_seqs = seed_pool_seqs[:,:,:,None]

negatives_fasta = fasta_path + fasta_diff_negatives 
negative_seqs = seq_functions.get_fasta_seqs(negatives_fasta)
negative_seqs = negative_seqs[:,:,:,None]

# combined. compute z scores and stds from this
combined_seqs = np.concatenate([negative_seqs,seed_pool_seqs])
zmeans = {}
zstds = {}
for ii in range(len(opt_models_names)):
    opt_models_name = opt_models_names[ii]
    zstds[opt_models_name] = np.std(opt_models[opt_models_name].predict(combined_seqs, batch_size=512)[:,0])
    zmeans[opt_models_name] = np.mean(opt_models[opt_models_name].predict(combined_seqs, batch_size=512)[:,0])


##bed with scores
bed_prefix = dh_prefix + 'data/tidy_data/celltype_specific_enhancers/bed/'
columns = ['chr', 'start', 'end', 'name', 'score','na0','na1','na2','na3','250']

diff_bed = bed_prefix + 'macaque_dh_celltypes_split_positives/rheMac10_' + target + 'vs' + avoid + '_positive/fold1/train.bed'
negatives_bed = bed_prefix + 'macaque_dh_celltypes_split_negatives/rheMac10_' + target + 'vs' + avoid + '_negative/fold1/train.bed'


seed_pool_data = []
with open(diff_bed) as f:
    for line in f:
        seed_pool_data.append(line.strip().split())
seed_pool_data = pd.DataFrame(seed_pool_data, columns=columns)
seed_pool_data['score'] = np.array(seed_pool_data['score'].values).astype(np.float64)

negative_data = []
with open(negatives_bed) as f:
    for line in f:
        negative_data .append(line.strip().split())
negative_data  = pd.DataFrame(negative_data , columns=columns)
negative_data['score'] = np.array(negative_data ['score'].values).astype(np.float64)

### get top specific sequences as initial seeds for mutations. start with highest scoring differential peaks
top_ids = seed_pool_data['score'].values.argsort()[::-1]
seed_seqs = seed_pool_seqs[top_ids[0:num_seed_seqs],:,:,:]
seed_seqs_data = seed_pool_data.iloc[top_ids[0:num_seed_seqs]]

bottom_ids = seed_pool_data['score'].values.argsort()
bottom_seqs = seed_pool_seqs[bottom_ids[0:num_seed_seqs],:,:,:]
bottom_seqs_data = seed_pool_data.iloc[bottom_ids[0:num_seed_seqs]]

# # correlation of all positives -- predictions and logFC
weights = np.ones(len(opt_models))

## compare logFC and x-axis
x = np.concatenate([seed_pool_seqs, negative_seqs])
x_data = pd.concat([seed_pool_data, negative_data], axis=0, ignore_index=True)


a=saturation_mutagenesis_functions.opt_models_predict(seqs=x, opt_models=opt_models, mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                       zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)
b = np.concatenate([negative_data['score'].values, seed_pool_data['score'].values])
spearman_corr, _ = spearmanr(a,b)
spearman_corr

a = saturation_mutagenesis_functions.opt_models_predict(seqs=seed_pool_seqs, opt_models=opt_models, mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                       zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)
b = seed_pool_data['score'].values
pearson_corr, _ = pearsonr(a,b)
pearson_corr

seqs_to_mutate = seed_seqs
initial_predictions_macaque_fold1=opt_models['macaque_' + target + 'vs' + avoid + '_fold1'].predict(seqs_to_mutate, batch_size=512)[:,0]

initial_predictions = saturation_mutagenesis_functions.opt_models_predict(seqs=seqs_to_mutate, opt_models=opt_models, mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                       zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)

bottom_predictions = saturation_mutagenesis_functions.opt_models_predict(seqs=bottom_seqs, opt_models=opt_models, mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                       zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

#Welch’s t-test
t_stat, p_val_t = ttest_ind(initial_predictions, bottom_predictions, equal_var=False)
print(f"Welch t-statistic = {t_stat:.3f}, p = {p_val_t:.3e}")

# 2) Mann–Whitney U test (nonparametric; tests for shift in distribution):
u_stat, p_val_u = mannwhitneyu(initial_predictions, bottom_predictions, alternative='two-sided')
print(f"Mann-Whitney U = {u_stat:.3f}, p = {p_val_u:.3e}")

####### initial gc content
initial_gc = seq_functions.compute_gc_content(seqs_to_mutate)
bottom_gc = seq_functions.compute_gc_content(bottom_seqs)

#####################
#####################
#### bayesian inference on initial and bottom
initial_bayesian_means = {}
initial_bayesian_stds = {}
for name, model in opt_models.items():
    dropout_preds = predict_with_uncertainty(
        model,
        seqs_to_mutate,   
        batch_size=512,
        num_trials=num_dropout_trials,
        return_trials=False  # we only need aggregates
    )
    # res["mean"] has shape [N, 1] for regression: grab the vector
    initial_bayesian_means[name] = dropout_preds["mean"].reshape(-1)
    initial_bayesian_stds[name] = dropout_preds["std"].reshape(-1)

bayesian_means_avged =  np.stack(list(initial_bayesian_means.values()), axis=0).mean(axis=0)
b = seed_pool_data['score'].values
pearson_corr, _ = pearsonr(initial_predictions,seed_seqs_data['score'].values)
#####################
#####################
### adaLead
# Import adalead from helpers module
from helpers import adalead_onehot as adalead
importlib.reload(saturation_mutagenesis_functions)
from saturation_mutagenesis_functions import opt_models_predict

adalead_obj = adalead(
    model_queries_per_batch= 3*num_seed_seqs + 1,
    eval_batch_size=num_seed_seqs,
    opt_models=opt_models,
    mean_type='arithmetic_min',
    alpha=0.5,
    weights=None,
    zmeans=zmeans,
    zstds=zstds,
    flip_positives=flip_positives
)

new_sequences_adalead = []
intermediate_adalead = [
    [None] * adalead_recombine_turns
    for _ in range(adalead_runs)
    ]

for adalead_run in range(adalead_runs):
    for adalead_turn in range(adalead_recombine_turns):
        print('run ' + str(adalead_run) + ': ' + 'turn ' + str(adalead_turn))
        if adalead_turn == 0:
            new_candidates = seqs_to_mutate.copy()
        new_candidates, predicted_fitnesses = adalead_obj.propose_sequences(new_candidates)
        
        intermediate_adalead[adalead_run][adalead_turn] = new_candidates.copy()

        print(predicted_fitnesses)
        ## rotate?
    new_sequence = new_candidates[np.argmax(predicted_fitnesses)]
    new_sequences_adalead.append(new_sequence)

new_adalead = np.stack(new_sequences_adalead, axis=0)

new_adalead_predictions = saturation_mutagenesis_functions.opt_models_predict(seqs=new_adalead, opt_models=opt_models, 
                                                            mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                                        zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)

new_adalead_testSet_predictions = testSet_model.predict(new_adalead, batch_size=512)[:,1]


new_adalead_gc = seq_functions.compute_gc_content(new_adalead)


#####################
#####################
#### saturation mutagenesis after adalead
import helpers as saturation_mutagenesis_functions
importlib.reload(saturation_mutagenesis_functions)
from saturation_mutagenesis_functions import saturation_mutagenesis_loop
t3 = time.time()
new_adalead_satmut, best_opt_fitness_vals, opt_fitness_vals, val_fitnesses_by_iter_dictionary,\
seqs_intermedate = saturation_mutagenesis_loop(seqs=new_adalead, opt_models=opt_models, iterations=satmut_iterations, 
    val_models = val_models, mean_type = 'arithmetic_min', alpha=0.5, predict_batch_size = 128,
    random_mode=False, return_intermediate_sequences=True, keep_parent = True, weights = None, zmeans=zmeans, zstds = zstds, flip_positives = flip_positives)

new_adalead_satmut_testSet_predictions = testSet_model.predict(new_adalead_satmut, batch_size=512)[:,1]

t4 = time.time()
print(t4-t3)
#############
#####################
#####################


new_adalead_satmut_gc = seq_functions.compute_gc_content(new_adalead_satmut)


#####################
#####################
#### bayesian inference
bayesian_means = {}
bayesian_stds = {}
randomMut_means = {}
randomMut_stds = {}
for name, model in opt_models.items():
    a, b = predict_with_random_mutations(
        model=model, 
        sequences=new_adalead_satmut,
        task='regression', 
        num_mutations=50, 
        num_trials=100)
    randomMut_means[name] = a
    randomMut_stds[name] = b
    dropout_preds = predict_with_uncertainty(
        model,
        new_adalead_satmut,   
        batch_size=512,
        num_trials=num_dropout_trials,
        return_trials=False  # we only need aggregates
    )
    # res["mean"] has shape [N, 1] for regression: grab the vector
    bayesian_means[name] = dropout_preds["mean"].reshape(-1)
    bayesian_stds[name] = dropout_preds["std"].reshape(-1)

z = opt_models['mouse_GLUT6vsGLUT5_fold1'].predict(new_adalead_satmut, batch_size=512)[:,0]
pearson_corr, _ = pearsonr(z, randomMut_stds['macaque_GLUT6vsGLUT5_fold1'])
pearson_corr
pearson_corr, _ = pearsonr(randomMut_stds['macaque_GLUT6vsGLUT5_fold1'], randomMut_means['macaque_GLUT6vsGLUT5_fold1'])
pearson_corr
#####################
#####################
#### random mutation



## hamming and levenshtein distances
importlib.reload(seq_functions)
hamming = seq_functions.avg_pairwise_hamming_onehot(new_adalead)

seq_functions.avg_pairwise_hamming_onehot(seqs_to_mutate)

max_different = seq_functions.max_pairwise_hamming_onehot(new_adalead_satmut)
hamming_max = seq_functions.avg_pairwise_hamming_onehot(max_different)
hamming_max

################ Save mutated sequences
################
################
synthetic_fasta_path = fasta_path + 'synthetic/' + target + 'vs' + avoid + '/'
os.makedirs(synthetic_fasta_path, exist_ok=True)

# initial seed sequences
tmp_seqs = seq_functions.batch_inverse_onehot(seqs_to_mutate)
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'initial_seeds_highFC_' + str(num_seed_seqs) + '_' + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

# bottom positive sequences
tmp_seqs = seq_functions.batch_inverse_onehot(bottom_seqs)
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'bottom_pos_sequences_lowFC_' + str(bottom_seqs.shape[0]) + '_' + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

seed_seqs = seed_pool_seqs[top_ids[0:num_seed_seqs],:,:,:]
seed_seqs_data = seed_pool_data.iloc[top_ids[0:num_seed_seqs]]

bottom_ids = seed_pool_data['score'].values.argsort()
bottom_seqs = seed_pool_seqs[bottom_ids[0:num_seed_seqs],:,:,:]



# after adalead runs
tmp_seqs = seq_functions.batch_inverse_onehot(new_adalead)
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'adalead_' + str(adalead_runs) + '_runs_' +  str(adalead_recombine_turns) + '_turns_' + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

# after 10 satmut
tmp_seqs = seq_functions.batch_inverse_onehot( seqs_intermedate[9] )
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'adalead_satmut_' + str(adalead_runs) + '_seqs_' +'10iters_'  + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

# after 20 satmut
tmp_seqs = seq_functions.batch_inverse_onehot( seqs_intermedate[19] )
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'adalead_satmut_' + str(adalead_runs) + '_seqs_' +'20iters_'  + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

################ Save all variables as session
################
################
session_dir = dh_prefix + 'data/tidy_data/synthetic_design/sessions/'
dill.dump_session(session_dir + target + 'vs' + avoid + '_satmut_adalead_session.pkl')


#### satmut of just skor2.103
####
####
####
skor2_103_satmut_iters = 50
skor2_103_satmut, _, _, skor2_103_val_fitnesses_by_iter_dictionary,\
skor2_103_seqs_intermedate = saturation_mutagenesis_loop(seqs=skor2_103_seq, opt_models=opt_models, iterations=skor2_103_satmut_iters, 
    val_models = val_models, mean_type = 'arithmetic_min', alpha=0.5, predict_batch_size = 128,
    random_mode=False, return_intermediate_sequences=True, keep_parent = True, weights = None, zmeans=zmeans, zstds = zstds, flip_positives = flip_positives)
importlib.reload(seq_functions)

# SKOR2.103 sequences to fasta
tmp_seqs = seq_functions.batch_inverse_onehot( skor2_103_seqs_intermedate )
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'Exc-SKOR2.103_satmut_' + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )


#### satmut of just skor2.55
####
####
####
skor2_55_satmut_iters = 50
skor2_55_satmut, _, _, skor2_55_val_fitnesses_by_iter_dictionary,\
skor2_55_seqs_intermedate = saturation_mutagenesis_loop(seqs=skor2_55_seq, opt_models=opt_models, iterations=skor2_55_satmut_iters, 
    val_models = val_models, mean_type = 'arithmetic_min', alpha=0.5, predict_batch_size = 128,
    random_mode=False, return_intermediate_sequences=True, keep_parent = True, weights = None, zmeans=zmeans, zstds = zstds, flip_positives = flip_positives)
importlib.reload(seq_functions)

# SKOR2.55 sequences to fasta
tmp_seqs = seq_functions.batch_inverse_onehot( skor2_55_seqs_intermedate )
str_dict = seq_dict = { str(i): seq for i, seq in enumerate(tmp_seqs) }
path = 'Exc-SKOR2.55_satmut_' + target + 'vs' + avoid + '.fa'
seq_functions.write_fasta(  str_dict  , synthetic_fasta_path + path   )

session_dir = dh_prefix + 'data/tidy_data/synthetic_design/sessions/'
dill.dump_session(session_dir + target + 'vs' + avoid + '_satmut_adalead_session_with_SKOR2.103_SKOR2.55.pkl')

long_run_iterations = 200
skor2_103_long_run, _, _, _,\
skor2_103_long_run_seqs_intermedate = saturation_mutagenesis_loop(seqs=skor2_103_seq, opt_models=opt_models, iterations=50, 
    val_models = val_models, mean_type = 'arithmetic_min', alpha=0.5, predict_batch_size = 512,
    random_mode=False, return_intermediate_sequences=True, keep_parent = True, weights = None, zmeans=zmeans, zstds = zstds, flip_positives = flip_positives)

skor2_103_satmut_25iters = skor2_103_seqs_intermedate[24]

original_skor2_103_testSet_predictions = testSet_model.predict(skor2_103_seq, batch_size=512)[:,1]
new_skor2_103_satmut_testSet_predictions = testSet_model.predict(skor2_103_satmut, batch_size=512)[:,1]
skor2_103_satmut_25iters_testSet_predictions = testSet_model.predict(skor2_103_satmut_25iters, batch_size=512)[:,1]

stack = np.concatenate(skor2_103_long_run_seqs_intermedate, axis=0)
skor2_103_satmut_long_run_testSet_predictions = testSet_model.predict(stack, batch_size=512)[:,1]


##### randomly tile TATA and ATATACA to see what happens to scores
def tile_motif_random(seq, motif, n):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = seq.shape[1]
    start = seq_len // 4
    end = 3 * seq_len // 4
    motif = motif.upper()
    inds = [mapping[c] for c in motif]
    m = len(inds)
    slots = np.arange(start, end - (m - 1), m)
    eye4 = np.eye(4, dtype=seq.dtype)
    result = [seq]
    for _ in range(n):
        arr = result[-1][0, :, :, 0].copy()
        pos = np.random.choice(slots)
        for i, b in enumerate(inds):
            arr[pos + i] = eye4[b]
        result.append(arr[np.newaxis, :, :, np.newaxis])
    return result

plt.rcParams.update({'font.size': 14})

stack_tata_middle = np.concatenate(skor2_103_tata_middle, axis=0)
skor2_103_tata_middle_testSet_predictions = testSet_model.predict(stack_tata_middle, batch_size=512)[:,1]

preds = saturation_mutagenesis_functions.opt_models_predict(seqs=stack, opt_models=opt_models, 
                                                            mean_type = 'arithmetic_min', alpha = 0.5, weights = None, 
                                        zmeans = zmeans, zstds = zstds, predict_batch_size = 512, flip_positives=flip_positives)
iters = np.arange(len(preds))
fig, ax = plt.subplots()
ax.plot(iters, preds, 'o', markersize=2, alpha = 0.5)  
ax.set_xlabel('Sat Mut Iteration')
ax.set_ylabel('Skor2.103 aggregate z-score')
plt.tight_layout()
plt.savefig(figurepath + '/' + target + 'vs' + avoid + 'zscore_skor2_103_long_run200.pdf')
plt.close()

fig, ax = plt.subplots()
ax.plot(iters, skor2_103_satmut_long_run_testSet_predictions, 'o', markersize=2, alpha = 0.5)  
ax.set_xlabel('Sat Mut Iteration')
ax.set_ylabel('Held Out Model Prediction')
plt.tight_layout()
plt.savefig(figurepath + '/' + target + 'vs' + avoid + 'testSet_preds_skor2_103_long_run200.pdf')
plt.close()

logits = np.log(skor2_103_satmut_long_run_testSet_predictions) - np.log1p(-skor2_103_satmut_long_run_testSet_predictions)
fig, ax = plt.subplots()
ax.plot(iters, logits , 'o', markersize=2, alpha = 0.5)  
ax.set_xlabel('Sat Mut Iteration')
ax.set_ylabel('Held Out Model Logit Prediction')
plt.tight_layout()
plt.savefig(figurepath + '/' + target + 'vs' + avoid + 'testSet_logits_skor2_103_long_run200.pdf')
plt.close()


skor2_103_tata_middle_testSet_predictions_runs = [testSet_model.predict(np.concatenate(tile_motif_random(
                                                  skor2_103_seq, motif= 'TATA', n = 50) ), batch_size=512)[:,1] for i in range(0,100)]
tata_preds = np.stack(skor2_103_tata_middle_testSet_predictions_runs, axis=0)
num_positions = tata_preds.shape[1]
tata_positions = np.arange(num_positions) * 4

skor2_103_atataca_middle_testSet_predictions_runs = [
    testSet_model.predict(
        np.concatenate(tile_motif_random(skor2_103_seq, motif='ATATACA', n=28)),
        batch_size=512
    )[:, 1]
    for i in range(100)
]
atataca_preds = np.stack(skor2_103_atataca_middle_testSet_predictions_runs, axis=0)

# compute positions separately for each motif
tata_num_positions = tata_preds.shape[1]
tata_positions = np.arange(tata_num_positions) * 4
atataca_num_positions = atataca_preds.shape[1]
atataca_positions = np.arange(atataca_num_positions) * 7

fig, ax = plt.subplots()

# Sat‑Mut scatter/line
satmut_line, = ax.plot(
    iters,
    skor2_103_satmut_long_run_testSet_predictions,
    'o',
    markersize=2,
    alpha=0.5,
    label='Saturation Mutagenesis'
)
# TATA median + IQR ribbon
tata_med = np.median(tata_preds, axis=0)
tata_q1, tata_q3 = np.percentile(tata_preds, [25, 75], axis=0)
tata_line, = ax.plot(
    tata_positions,
    tata_med,
    '-',
    linewidth=1.5,
    color='C1',
    label='TATA Implant median'
)
ax.fill_between(
    tata_positions,
    tata_q1,
    tata_q3,
    color='C1',
    alpha=0.3,
    label='TATA Implant IQR'
)
# ATATACA median + IQR ribbon
atataca_med = np.median(atataca_preds, axis=0)
atataca_q1, atataca_q3 = np.percentile(atataca_preds, [25, 75], axis=0)
atataca_line, = ax.plot(
    atataca_positions,
    atataca_med,
    '-',
    linewidth=1.5,
    color='C2',
    label='ATATACA Implant median'
)
ax.fill_between(
    atataca_positions,
    atataca_q1,
    atataca_q3,
    color='C2',
    alpha=0.3,
    label='ATATACA Implant IQR'
)
ax.set_xlabel('# Nucleotides Mutated')
ax.set_ylabel('Held‑Out Model Prediction')
ax.legend()
plt.tight_layout()
plt.savefig(f"{figurepath}/{target}vs{avoid}_testSet_preds_longrun200_and_TATA_ATATACA.pdf")
plt.close()


### mean and standard deviation of predictions with random mutations
importlib.reload(saturation_mutagenesis_functions)
from saturation_mutagenesis_functions import predict_with_random_mutations
a,b = predict_with_random_mutations(model=testSet_model, sequences=skor2_103_satmut_25iters, task='classification', num_mutations=100, num_trials=100)
c,d = predict_with_random_mutations(model=testSet_model, sequences=skor2_103_seq, task='classification', num_mutations=100, num_trials=100)


skor2_103_original_gc = seq_functions.compute_gc_content(skor2_103_seq)
skor2_103_25iters_gc = seq_functions.compute_gc_content(skor2_103_satmut_25iters)
skor2_103_replace_AT_mutant = seq_functions.mutate_onehot(skor2_103_satmut_25iters, target_bases=['A','T'], new_bases=['G','C'], num_replacements=100)
skor2_103_random_mutant = seq_functions.mutate_onehot(skor2_103_satmut_25iters, target_bases=['A','T','G','C'], new_bases=['A','T','G','C'], num_replacements=25)
skor2_103_replace_AT_mutant_testSet_predictions = testSet_model.predict(skor2_103_replace_AT_mutant, batch_size=512)[:,1]
skor2_103_random_mutant_testSet_predictions = testSet_model.predict(skor2_103_random_mutant, batch_size=512)[:,1]

### towards skor2 targeting
long_run_skor2_103_satmut_dna = seq_functions.batch_inverse_onehot(skor2_103_long_run)[0]
skor2_103_satmut_dna = seq_functions.batch_inverse_onehot(skor2_103_satmut)[0]
skor2_103_dna = seq_functions.batch_inverse_onehot(skor2_103_seq)[0]
assert len(skor2_103_dna) == len(skor2_103_satmut_dna), "Sequences must be equal length"
# Find all differing positions
diffs = [
    (i, skor2_103_dna[i], skor2_103_satmut_dna[i])
    for i in range(len(skor2_103_dna))
    if skor2_103_dna[i] != skor2_103_satmut_dna[i]
]
# Print a summary
print(f"Found {len(diffs)} mutations:")
for pos, before, after in diffs:
    # use pos+1 if you prefer 1-based indexing
    print(f"  Position {pos}: {before} → {after}")

### towards LMO3 targeting
switch_to_lmo3_dna = seq_functions.batch_inverse_onehot(switch_to_lmo3)[0]
# Find all positions where they differ
diffs = [
    (i, skor2_103_dna[i], switch_to_lmo3_dna[i])
    for i in range(len(skor2_103_dna))
    if skor2_103_dna[i] != switch_to_lmo3_dna[i]
]
# Print the summary
print(f"Found {len(diffs)} mutations:")
for pos, before, after in diffs:
    # use pos+1 if you prefer 1-based indexing
    print(f"  Position {pos}: {before} → {after}")