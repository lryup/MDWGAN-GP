from utils import *
from tf_utils import *
from graph_function import *  #come from   https://github.com/juexinwang/scGNN
from scipy.linalg import fractional_matrix_power
from dag import DAG #come from https://github.com/tntrung/dag-gans
import numpy as np

EPOCHS = 500
BATCH_SIZE = 32
LATENT_DIM = 10
MODELS_DIR = 'checkpoints/models/'
calculate_gcn=True#
gcn_c = True  # whether to use gcn correlation
gcn_e = True  # whether to use gcn euclidean

if __name__ == '__main__':
    # GPU limit
    limit_gpu()

    # Constants
    root_gene = 'CRP'  # Set to 'CRP' to select the CRP hierarchy. Set to None to use full set of genes
    minimum_evidence = 'weak'
    max_depth = np.inf
    retrain = False
    interactions = None

    # Load data
    expr, gene_symbols, sample_names = load_ecoli(root_gene=root_gene,
                                                  minimum_evidence=minimum_evidence,
                                                  max_depth=max_depth)

    # Load conditions
    encoded_conditions = None
    vocab_dicts = None

    _, encoded_conditions, vocab_dicts = ecoli_discrete_conditions(sample_names)
    print('Conditions: ', encoded_conditions)
    encoded_conditions = np.int32(encoded_conditions)

    # encoded_conditions=np.zeros((907,7))
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    print('Filename: ', file_name)

    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test_v3(sample_names,train_rate=0.1)
    expr = np.float32(expr)
    expr_train = expr[train_idxs, :]
    expr_test = expr[test_idxs, :]
    cond_train = encoded_conditions[train_idxs, :]
    cond_test = encoded_conditions[test_idxs, :]

    # Normalise data
    x_mean = np.mean(expr_train, axis=0)
    x_std = np.std(expr_train, axis=0)
    expr_train = standardize(expr_train, mean=x_mean, std=x_std)
    expr_test = standardize(expr_test, mean=x_mean, std=x_std)
    print('expr_train.shape',expr_train.shape)
    print('expr_test.shape',expr_test.shape)

    #add gcn
    x_train=expr_train
    x_test=expr_test
    if (calculate_gcn):
        useGAEembedding = True
        adj, edgeList = generateAdj(x_train, graphType='KNNgraphStatsSingleThread', para='correlation' +
                                                                                         ':' + str(10),
                                    adjTag=(useGAEembedding or False))
        print("...calculate the sample correclation adj")
        adj_m = adj.toarray()
        A = adj_m
        # A = penalty_degree(nb_percent, A)
        D = np.diag(A.sum(axis=1))
        I = np.array(np.eye(A.shape[0]))
        A_hat = A + I
        D_hat = np.array(np.sum(A_hat, axis=1))
        # D_hat.shape
        D_hat = np.array(np.diag(D_hat))
        A_norm = np.dot(np.dot(fractional_matrix_power(D_hat, -0.5), A_hat), fractional_matrix_power(D_hat, -0.5))
        adj_cor = np.dot(A_norm, x_train)
        x_train_gcn = adj_cor

    if (calculate_gcn):
        #         print("...calculate the euclidean ")
        useGAEembedding = True
        adj, edgeList = generateAdj(x_train, graphType='KNNgraphStatsSingleThread', para='euclidean' +
                                                                                         ':' + str(10),
                                    adjTag=(useGAEembedding or False))
        print("...calculate the sample euclidean ppi")
        adj_m = adj.toarray()
        A = adj_m
        # A = penalty_degree(nb_percent, A)
        D = np.diag(A.sum(axis=1))
        I = np.array(np.eye(A.shape[0]))
        A_hat = A + I
        D_hat = np.array(np.sum(A_hat, axis=1))
        # D_hat.shape
        D_hat = np.array(np.diag(D_hat))
        A_norm = np.dot(np.dot(fractional_matrix_power(D_hat, -0.5), A_hat), fractional_matrix_power(D_hat, -0.5))
        adj_euc = np.dot(A_norm, x_train)
        x_train_gcn2 = adj_euc

    # Define model
    vocab_sizes = [len(c) for c in vocab_dicts]
    # vocab_sizes=[1]
    print('Vocab sizes: ', vocab_sizes)
    num_covs_train = np.zeros(shape=[cond_train.shape[0], 1])  # Dummy
    num_covs_test = np.zeros(shape=[cond_test.shape[0], 1])  # Dummy
    nb_numeric = num_covs_train.shape[-1]
    x_dim = expr_train.shape[-1]

    x_dim_gcn = x_train_gcn.shape[-1]

    gen = make_generator(x_dim, vocab_sizes, nb_numeric, z_dim=LATENT_DIM, h_dims=[128, 128])
    #notice:  The  rotation default value is 0(No rotation)
    dag = DAG(D_loss_func, G_loss_func, policy=['rotation'], policy_weight=[1.0])
    n_augments = dag.get_n_augments_from_policy()

    disc = make_discriminator_gene_dag(x_dim, vocab_sizes, nb_numeric, n_augments,
                                       h_dims=[128, 128])

    # Evaluation metrics
    def score_fn_gcn_vae(x_test, cat_covs_test, num_covs_test):
        def _score(gen):
            x_gen = predict(cc=cat_covs_test,
                            nc=num_covs_test,
                            gen=gen)

            gamma_dx_dz = gamma_coef(x_test, x_gen)
            return gamma_dx_dz
            # score = (x_test - x_gen) ** 2
            # return -np.mean(score)

        return _score


    # Function to save models;cor and euc,no_cor_euc
    def save_fn(models_dir=MODELS_DIR):
        gen.save(models_dir + 'better_MDWGAN_ecoli_model.h5')
    #lry add
    def save_fn_last(models_dir=MODELS_DIR):
        print("The model stops when it reaches the number of patience. If you want to save the last model, comment break in patience.")
        # gen.save(models_dir + 'last_MDWGAN_epoch_500.h5')

    train_gcn_gene_dag(gcn_c=gcn_c, gcn_e=gcn_e, dataset=x_train,  x_train_gcn=x_train_gcn, x_train_gcn2=x_train_gcn2,
                                     patience=20,
                                     cat_covs=cond_train,
                                     num_covs=num_covs_train,
                                     z_dim=LATENT_DIM,
                                     batch_size=BATCH_SIZE,
                                     epochs=EPOCHS,
                                     # nb_critic=CONFIG['nb_critic'],
                                     gen=gen,
                                     disc=disc,
                                     p_aug=0.25,
                                     # gen_opt=gen_opt,
                                     # disc_opt=disc_opt,
                                     score_fn=score_fn_gcn_vae(x_test, cond_test, num_covs_test),
                                     save_fn=save_fn,
                                     save_fn_last=save_fn_last)
    score = score_fn_gcn_vae(x_test, cond_test, num_covs_test)(gen)
    print('Gamma(Dx, Dz): {:.2f}'.format(score))
