import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import os
import datetime
import numpy as np
from dag import DAG
tfk = tf.keras
tfkl = tf.keras.layers


CHECKPOINTS_DIR = '../checkpoints/'


# ------------------
# LIMIT GPU USAGE
# ------------------

def limit_gpu(gpu_idx=0, mem=2 * 1024):
    """
    Limits gpu usage
    :param gpu_idx: Use this gpu
    :param mem: Maximum memory in bytes
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # Use a single gpu
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

            # Limit memory
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem)])  # 2 GB
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def make_generator_tanh(x_dim, vocab_sizes, nb_numeric, h_dims=None, z_dim=10):
    """
    Make generator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :param z_dim: Number of input units
    :return: generator
    """
    # Define inputs
    z = tfkl.Input((z_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []
    total_emb_dim = 0

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)
        total_emb_dim += emb_dim
    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])
    total_emb_dim += nb_numeric

    def make_generator_emb(x_dim, emb_dim, h_dims=None, z_dim=10):
        if h_dims is None:
            # h_dims = [256, 256]
            #lry change
            h_dims = [128,256, 512]

        z = tfkl.Input((z_dim,))
        t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
        h = tfkl.Concatenate(axis=-1)([z, t_emb])
        for d in h_dims:
            h = tfkl.Dense(d)(h)
            h = tfkl.ReLU()(h)
        h = tfkl.Dense(x_dim,activation='tanh')(h)
        model = tfk.Model(inputs=[z, t_emb], outputs=h)
        return model

    gen_emb = make_generator_emb(x_dim=x_dim,
                                 emb_dim=total_emb_dim,
                                 h_dims=h_dims,
                                 z_dim=z_dim)
    model = tfk.Model(inputs=[z, cat, num], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model


def make_generator(x_dim, vocab_sizes, nb_numeric, h_dims=None, z_dim=10):
    """
    Make generator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :param z_dim: Number of input units
    :return: generator
    """
    # Define inputs
    z = tfkl.Input((z_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []
    total_emb_dim = 0

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)
        total_emb_dim += emb_dim
    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])
    total_emb_dim += nb_numeric

    def make_generator_emb(x_dim, emb_dim, h_dims=None, z_dim=10):
        if h_dims is None:
            h_dims = [256, 256]
            #lry change
            # h_dims = [128,256, 1024]

        z = tfkl.Input((z_dim,))
        t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
        h = tfkl.Concatenate(axis=-1)([z, t_emb])
        for d in h_dims:
            h = tfkl.Dense(d)(h)
            h = tfkl.ReLU()(h)
        h = tfkl.Dense(x_dim)(h)
        model = tfk.Model(inputs=[z, t_emb], outputs=h)
        return model

    gen_emb = make_generator_emb(x_dim=x_dim,
                                 emb_dim=total_emb_dim,
                                 h_dims=h_dims,
                                 z_dim=z_dim)
    model = tfk.Model(inputs=[z, cat, num], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model

def D_loss_func(x_real, x_fake,cc,nc, netD,grad_penalty_weight=10, dag=False, dag_idx=0):
    if dag==False:
        d_real, _ = netD([x_real, cc, nc], training=True)
        d_fake, _ = netD([x_fake, cc, nc], training=True)
       # d_real,_ = netD(x_real)
       # d_fake,_ = netD(x_fake)
    else:
#         import pdb
#         pdb.set_trace()
        _, d_reals = netD([x_real,cc,nc],training=True)
#         d_real = d_reals[dag_idx]#因为只返回第一个，估计不对
        d_real = d_reals
        _, d_fakes = netD([x_fake, cc, nc], training=True)
#         d_fake = d_fakes[dag_idx]
        d_fake = d_fakes
    # d_cost = tf.reduce_mean(tf.nn.softplus(d_real)) + tf.reduce_mean(tf.nn.softplus(-d_fake))
    d_cost = discriminator_loss(d_real, d_fake) \
                    + grad_penalty_weight * calc_gradient_penalty(lambda x_real: netD([x_real, cc, nc], training=True), x_real, x_fake,dag)

    return d_cost
def G_loss_func(x_real, x_fake, cc,nc,netD, dag=False, dag_idx=0):
    if dag==False:
        d_fake,_= netD([x_fake, cc, nc], training=False)
    else:
        _, d_fakes = netD([x_fake, cc, nc], training=False)
#         d_fake = d_fakes[dag_idx]
        d_fake = d_fakes


    # Compute losses
    g_cost = generator_loss(d_fake)
    # g_cost = tf.reduce_mean(tf.nn.softplus(-d_fake))
    return g_cost
def make_discriminator_dag(x_dim, vocab_sizes, nb_numeric,n_augments, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
        # h_dims = [256, 256]
        #lry change
        h_dims = [512, 256,128]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    h = tfkl.Dense(1)(h)
    #lry add dag
    outputs_dag = []
    for i in range(n_augments):
        outputs_dag.append(h)
        outputs_dag = tf.concat(outputs_dag, 0)
    model = tfk.Model(inputs=[x, cat, num], outputs=[h,outputs_dag])
    return model
def make_discriminator_gene_dag(x_dim, vocab_sizes, nb_numeric,n_augments, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
        # h_dims = [256, 256]
        #lry change
        h_dims = [512, 256,128]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    out = tfkl.Dense(1)(h)
    #lry add dag
    outputs_dag = []
    for i in range(n_augments):
        outputs_dag.append(tfkl.Dense(1)(h))
        outputs_dag = tf.concat(outputs_dag, 0)
    model = tfk.Model(inputs=[x, cat, num], outputs=[out,outputs_dag])
    return model
def make_discriminator(x_dim, vocab_sizes, nb_numeric, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
#         h_dims = [256, 256]
        #lry change
        h_dims = [1024, 256,128]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    h = tfkl.Dense(1)(h)
    model = tfk.Model(inputs=[x, cat, num], outputs=h)
    return model
def make_discriminator_cf(x_dim, vocab_sizes, nb_numeric, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
#         h_dims = [256, 256]
        #lry change
        h_dims = [1024,512, 1024]

#     x = tfkl.Input((x_dim,))
    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    h = tfkl.Dense(1)(h)
    model = tfk.Model(inputs=[x, cat, num], outputs=h)
    return model
def make_discriminator_gcn_vae(x_dim, vocab_sizes, nb_numeric, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
        h_dims = [256, 256]
        #lry change
        # h_dims = [1024, 256,128]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    lh = tfkl.Dense(1)(h)
    model = tfk.Model(inputs=[x, cat, num], outputs=[h,lh])
    return model

def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss
    """
    return tf.reduce_mean(y_true * y_pred)


def generator_loss(fake_output):
    """
    Generator loss
    """
    return wasserstein_loss(-tf.ones_like(fake_output), fake_output)


def gradient_penalty(f, real_output, fake_output):
    """
    Gradient penalty of WGAN-GP
    :param f: discriminator function without sample covariates as input
    :param real_output: real data
    :param fake_output: fake data
    :return: gradient penalty
    """
    alpha = tf.random.uniform([real_output.shape[0], 1], 0., 1.)
    diff = fake_output - real_output
    inter = real_output + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))  # real_output
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp

def calc_gradient_penalty(f, real_output, fake_output,dag=False):
    """
    Gradient penalty of WGAN-GP
    :param f: discriminator function without sample covariates as input
    :param real_output: real data
    :param fake_output: fake data
    :return: gradient penalty
    """
    alpha = tf.random.uniform([real_output.shape[0], 1], 0., 1.)
    diff = fake_output - real_output
    inter = real_output + (alpha * diff)
#     with tf.GradientTape() as t:
#         t.watch(inter)
#         pred = f(inter)
    #lry change
    if dag==False:
        with tf.GradientTape() as t:
            t.watch(inter)
            pred,_ = f(inter)
    else:
        with tf.GradientTape() as t:
            t.watch(inter)
            _,pred = f(inter)   
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))  # real_output
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp
def discriminator_loss(real_output, fake_output):
    """
    Critic loss
    """
    real_loss = wasserstein_loss(-tf.ones_like(real_output), real_output)
    fake_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
#lry change
# @tf.Session
def train_disc(x, z, cc, nc, gen, disc, disc_opt, grad_penalty_weight=10, p_aug=0, norm_scale=0.5):
    """
    Train critic
    :param x: Batch of expression data
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param disc_opt: Critic optimizer
    :param grad_penalty_weight: Weight for the gradient penalty
    :return: Critic loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=False)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
        x = x + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=True)
        disc_real = disc([x, cc, nc], training=True)

        # Compute losses
        disc_loss = discriminator_loss(disc_real, disc_out) \
                    + grad_penalty_weight * gradient_penalty(lambda x: disc([x, cc, nc], training=True), x, x_gen)

    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return disc_loss

#lry add function
#about function see
# https://www.jianshu.com/p/47172eb86b39
def gan_loss(logits, is_real=True):
    """Computes standard gan loss between logits and labels

        Arguments:
            logits {[type]} -- output of discriminator

        Keyword Arguments:
            isreal {bool} -- whether labels should be 0 (fake) or 1 (real) (default: {True})
        """
    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )
def train_disc_gcn_gene_dag_c_e(gcn_c,gcn_e,x, z, x_gcn,x_gcn2,   cc, nc, gen, disc, disc_opt, grad_penalty_weight=10, p_aug=0, norm_scale=0.5):
    """
    Train critic
    :param x: Batch of expression data
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param disc_opt: Critic optimizer
    :param grad_penalty_weight: Weight for the gradient penalty
    :return: Critic loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass
        
        x_gen = gen([z, cc, nc], training=False)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
        x = x + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
        gcn_c_loss=0.0
        gcn_e_loss=0.0
        disc_loss_r_z = D_loss_func(x, x_gen, cc, nc, disc, grad_penalty_weight)
#         disc_loss_r_z=0.0
        if(gcn_c+gcn_e>0):
            dag = DAG(D_loss_func, G_loss_func, policy=['trans'], policy_weight=[1.0])#policy=['rotation'];lry change trans,no use
        #之前分开考虑，后面发现dag,disc输出是一样的，不用，所以后面0.2只要x_gcn_gen即可
            if(gcn_c):
    #             print('dis gcn string')
    #             x_gcn_gen=gen_gcn([x_gcn,cc,nc],training=False)
                x_gen1=x_gen+ augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
                x_gcn_gen=x_gcn#按理来说，这里应该将z进行变化，但变化gcn,你是一个一个变换的，比较费时，不太可能，先试生成的
                x_gcn_gen = x_gcn_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
    #             import pdb
    #             pdb.set_trace()
                gcn_c_loss=dag.compute_discriminator_loss(x_gcn_gen,x_gen1,cc,nc,disc,grad_penalty_weight)
                # print('train:gcn_c_loss:',gcn_c_loss)
            if(gcn_e):
    #             print('dis gcn h3')
                x_gen2 = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
                x_gcn_gen2=x_gcn2
                x_gcn_gen2 = x_gcn_gen2 + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
                gcn_e_loss=dag.compute_discriminator_loss(x_gcn_gen2,x_gen2,cc,nc,disc,grad_penalty_weight)
                # print('train:gcn_e_loss:', gcn_e_loss)
            disc_loss = disc_loss_r_z+0.2*((gcn_c_loss+gcn_e_loss)/(gcn_c+gcn_e))#计算两个的调控的均值True+True=2;true+false=1

        else:
            disc_loss = disc_loss_r_z
        # disc_loss=disc_loss+disc_real_loss+disc_fake_loss
    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return disc_loss

def train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=0, norm_scale=1):
    """
    Train generator
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param gen_opt: Generator optimiser
    :return: Generator loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as gen_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=True)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=False)

        # Compute losses
        gen_loss = generator_loss(disc_out)
        # print('ws:', gen_loss)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))

    return gen_loss

def fold_change1(x):
    """
    Calcute two fold change
    """
    x_mean = tf.math.reduce_mean(x, axis=1)
    x_std = tf.math.reduce_std(x, axis=1)
    var_coef = x_std / x_mean
    var = x_mean + var_coef * x_mean ** 2
    var_log = tf.math.log(var + 1)
    return var_log
def coef_var(x):
    """
    Calcute two fold change
    """
    x_mean = tf.math.reduce_mean(x, axis=1)
    x_std = tf.math.reduce_std(x, axis=1)
    var_coef = x_std / x_mean
    # var = x_mean + var_coef * x_mean ** 2
    # var_log = tf.math.log(var + 1)
    return var_coef

    var_log = tf.math.log(var + 1)
    return var_log
def coef_var_gene(x):
    """
    Calcute two fold change
    """
    x_mean = tf.math.reduce_mean(x, axis=0)
    x_std = tf.math.reduce_std(x, axis=0)
    #because nan
    mask_value=0
    mask = tf.not_equal(x_mean, tf.constant(mask_value, dtype=x_mean.dtype))
    paddings = tf.ones_like(x_mean) * 1e-8
    out = tf.where(mask, x_mean, paddings)
    var_coef = x_std / out
    var = x_mean + var_coef * x_mean ** 2
    var_log = tf.math.log(var + 0.01)
    return var_coef
def fold_change(x):
    """
    Calcute two fold change
    """
    x_mean = tf.math.reduce_mean(x, axis=0)
    return x_mean
def normalize(data):
    minx=tf.reduce_min(data)
    maxx=tf.reduce_max(data)
    return (data-minx)/(maxx-minx)

def train_gen_mean_gene_dag_c_e(gcn_c,gcn_e,x,  x_gcn,x_gcn2,z, cc, nc, gen, disc,
                  gen_opt, p_aug=0, norm_scale=1):
    """
    Train generator
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param gen_opt: Generator optimiser
    :return: Generator loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as gen_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=True)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
        x = x + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
        gen_loss=G_loss_func(x,x_gen,cc,nc,disc)
        dag = DAG(D_loss_func, G_loss_func, policy=['rotation'], policy_weight=[1.0])
        gcn_c_loss=0.0
        gcn_e_loss=0.0
        if (gcn_c + gcn_e > 0):
            if(gcn_c):
    #             print('gen gcn string')
                x_gen1 = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
                x_gcn_gen=x_gcn#gen_gcn([x_gcn,cc,nc],training=False)
                x_gcn_gen = x_gcn_gen + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
                gcn_c_loss=dag.compute_generator_loss(x_gcn_gen,x_gen1,cc,nc,disc)
                # print('gcn_c_loss:',gcn_c_loss)
            if(gcn_e):
    #             print('gen gcn h3')
                x_gen2 = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))
                x_gcn_gen2=x_gcn2#gen_gcn([x_gcn2,cc,nc],training=False)
                x_gcn_gen2 = x_gcn_gen2 + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
                gcn_e_loss=dag.compute_generator_loss(x_gcn_gen2,x_gen2,cc,nc,disc)
                # print('gcn_e_loss:', gcn_e_loss)
            gen_loss = gen_loss + 0.2 * ( (gcn_c_loss + gcn_e_loss) / (gcn_c + gcn_e))  # 计算两个的调控的均值True+True=2;true+false=1

        else:
            gen_loss=gen_loss

    # gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    # gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
    #lry change
    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))

    return gen_loss


def train(dataset, cat_covs, num_covs, z_dim, epochs, batch_size, gen, disc, score_fn, save_fn,save_fn_last_raw,
          gen_opt=None, disc_opt=None, nb_critic=5, verbose=True, checkpoint_dir='./checkpoints/cpkt',
          log_dir='./logs/', patience=10, p_aug=0, norm_scale=0.5):
    """
    Train model
    :param dataset: Numpy matrix with data. Shape=(nb_samples, nb_genes)
    :param cat_covs: Categorical covariates. Shape=(nb_samples, nb_cat_covs)
    :param num_covs: Numerical covariates. Shape=(nb_samples, nb_num_covs)
    :param z_dim: Int. Latent dimension
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param gen: Generator model
    :param disc: Critic model
    :param gen_opt: Generator optimiser
    :param disc_opt: Critic optimiser
    :param score_fn: Function that computes the score: Generator => score.
    :param save_fn:  Function that saves the model.
    :param nb_critic: Number of critic updates for each generator update
    :param verbose: Print details
    :param checkpoint_dir: Where to save checkpoints
    :param log_dir: Where to save logs
    :param patience: Number of epochs without improving after which the training is halted
    """
    # Optimizers
    if gen_opt is None:
        gen_opt = tfk.optimizers.RMSprop(5e-4)
    if disc_opt is None:
        disc_opt = tfk.optimizers.RMSprop(5e-4)

    # Set up logs and checkpoints
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    gen_log_dir = log_dir + current_time + '/gen'
    disc_log_dir = log_dir + current_time + '/disc'
    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    gen_losses = tfk.metrics.Mean('gen_loss', dtype=tf.float32)
    disc_losses = tfk.metrics.Mean('disc_loss', dtype=tf.float32)
    best_score = -np.inf
    initial_patience = patience

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            x = dataset[i: i + batch_size, :]
            cc = cat_covs[i: i + batch_size, :]
            nc = num_covs[i: i + batch_size, :]

            # Train critic
            disc_loss = None
            for _ in range(nb_critic):
                z = tf.random.normal([x.shape[0], z_dim])
                disc_loss = train_disc(x, z, cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
            disc_losses(disc_loss)

            # Train generator
            z = tf.random.normal([x.shape[0], z_dim])
            gen_loss = train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            #lry change loss
            # gen_loss = lry_train_gen(x,z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            gen_losses(gen_loss)

        # Logs
        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', disc_losses.result(), step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', gen_losses.result(), step=epoch)

        # Save the model
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            score = score_fn(gen)
            if score > best_score:
                print('Saving model ...')
                save_fn()
                best_score = score
                patience = initial_patience
            else:
                patience -= 1
                print('this is all raw last save,is bed')
                save_fn_last_raw()


            if verbose:
                print('Score: {:.3f}'.format(score))

        if verbose:
            print('Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}'.format(epoch + 1,
                                                                         gen_losses.result(),
                                                                         disc_losses.result()))
        gen_losses.reset_states()
        disc_losses.reset_states()

        if patience == 0:
            print('Patience is at its maximum, if you not want the last model, please remove the comment "break".')
            # break
# import torch
import random
#np
def negative_sampling(s_zr,s_pm, items):
    zr_all, pm_all = [], []
    for i in range(items.shape[0]):
#         import pdb
#         pdb.set_trace()
        where_zeros = np.where(items[i] == 0)[0].tolist()
        n = round(len(where_zeros) * s_zr) if isinstance(s_zr, float) else s_zr
        zr_pos = random.sample(where_zeros, n)
        zr = np.zeros_like(items[i])
        zr[zr_pos] = 1
        zr_all.append(zr)

        n = round(len(where_zeros) * s_pm) if isinstance(s_pm, float) else s_pm
        pm_pos = random.sample(where_zeros, n)
        pm = np.zeros_like(items[i])
        pm[pm_pos] = 1
        pm_all.append(pm)

    return np.stack(zr_all, axis=0), np.stack(pm_all, axis=0)
            


def train_gcn_gene_dag(gcn_c,gcn_e,dataset, x_train_gcn,x_train_gcn2, cat_covs, num_covs, z_dim, epochs, batch_size, gen, disc, score_fn, save_fn,save_fn_last,
          gen_opt=None, disc_opt=None, nb_critic=5, verbose=True, checkpoint_dir='./checkpoints/cpkt',
          log_dir='./logs/', patience=10, p_aug=0, norm_scale=0.5):
    """
    Train model
    :param dataset: Numpy matrix with data. Shape=(nb_samples, nb_genes)
    :param cat_covs: Categorical covariates. Shape=(nb_samples, nb_cat_covs)
    :param num_covs: Numerical covariates. Shape=(nb_samples, nb_num_covs)
    :param z_dim: Int. Latent dimension
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param gen: Generator model
    :param disc: Critic model
    :param gen_opt: Generator optimiser
    :param disc_opt: Critic optimiser
    :param score_fn: Function that computes the score: Generator => score.
    :param save_fn:  Function that saves the model.
    :param nb_critic: Number of critic updates for each generator update
    :param verbose: Print details
    :param checkpoint_dir: Where to save checkpoints
    :param log_dir: Where to save logs
    :param patience: Number of epochs without improving after which the training is halted
    """
    # Optimizers
    if gen_opt is None:
        gen_opt = tfk.optimizers.RMSprop(5e-4)
    if disc_opt is None:
        disc_opt = tfk.optimizers.RMSprop(5e-4)

    # Set up logs and checkpoints
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    gen_log_dir = log_dir + current_time + '/gen'
    disc_log_dir = log_dir + current_time + '/disc'
    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    gen_losses = tfk.metrics.Mean('gen_loss', dtype=tf.float32)
    disc_losses = tfk.metrics.Mean('disc_loss', dtype=tf.float32)
    #lry add
    mse_losses=tfk.metrics.Mean('mse_fake_rale',dtype=tf.float32)
    gcn_losses = tfk.metrics.Mean('gcn_fake_rale', dtype=tf.float32)
    best_score = -np.inf
    initial_patience = patience

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            x = dataset[i: i + batch_size, :]
            cc = cat_covs[i: i + batch_size, :]
            nc = num_covs[i: i + batch_size, :]

            # #lry add gcn
            x_gcn=x_train_gcn[i:i+batch_size,:]
            x_gcn2=x_train_gcn2[i:i+batch_size,:]
            # Train critic
            disc_loss = None
            for _ in range(nb_critic):
                z = tf.random.normal([x.shape[0], z_dim])
                # disc_loss = train_disc(x, z, cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
                # # #lry add gcn
                # disc_loss_gcn=train_disc(x_gcn, z, cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
                # disc_loss=disc_loss_gcn
                # disc_loss=disc_loss+disc_loss_gcn

                # lry change gcn
#                 x_gcn=x_gcn
                #lry_train_disc_gcn_gene_dag_string_h3
                disc_loss = train_disc_gcn_gene_dag_c_e(gcn_c,gcn_e,x, z, x_gcn,x_gcn2,   cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
            disc_losses(disc_loss)

            # Train generator
            z = tf.random.normal([x.shape[0], z_dim])
            #lry change
            # z=x_gcn
            # gen_loss = train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            #lry change loss
#             gen_loss,mse_loss = lry_train_gen_z(x,x_train_max,x_train_min,alpha,meau,x_mean,x_std,x_gcn,gen_gcn,x_train_mean,   z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
#             gen_loss,mse_loss = lry_train_gen_l2(x,x_train_max,x_train_min,alpha,meau,x_mean,x_std,x_gcn,gen_gcn,x_train_mean,   z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)#不好，比较集中
            gen_loss = train_gen_mean_gene_dag_c_e(gcn_c,gcn_e,x,x_gcn,x_gcn2,  z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)

#             mse_loss = lry_train_gen_mean_only_h3(x,x_train_max,x_train_min,alpha,meau,x_mean,x_std,x_gcn,x_gcn2,gen_gcn,x_train_mean,x_train_qut_dw,x_train_qut_up,   z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            gen_losses(gen_loss)
#             gcn_loss= lry_train_gen_gcn_string_h3(gcn_string,gcn_h3,x,x_train_max,x_train_min,alpha,meau,x_mean,x_std,x_gcn,x_gcn2,gen_gcn,x_train_mean,   z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)


        # Logs
        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', disc_losses.result(), step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', gen_losses.result(), step=epoch)

        # Save the model
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            score = score_fn(gen)
            if score > best_score:
                print('Saving model ...')
                save_fn()
                best_score = score
                patience = initial_patience
            else:
                patience -= 1
                print('Have run the countdown patiently {:.2f} times, score has not improved. '.format(patience))
                save_fn_last()

            if verbose:
                print('Score: {:.3f}'.format(score))

        if verbose:
            print('Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}'.format(epoch + 1,
                                                                         gen_losses.result(),
                                                                         disc_losses.result()
                                                                        ))
        gen_losses.reset_states()
        disc_losses.reset_states()
        #lry add
        mse_losses.reset_states()
        gcn_losses.reset_states()

        if patience == 0:
            print('this run all expr,already is best')
            # break

#             break

def predict(cc, nc, gen, z=None, training=False):
    """
    Make predictions
    :param cc: Categorical covariates
    :param nc: Numerical covariates
    :param gen: Generator model
    :param z: Latent input
    :param training: Whether training
    :return: Sampled data
    """
    nb_samples = cc.shape[0]
    if z is None:
        z_dim = gen.input[0].shape[-1]
        z = tf.random.normal([nb_samples, z_dim])
    out = gen([z, cc, nc], training=training)
    if not training:
        return out.numpy()
    return out
