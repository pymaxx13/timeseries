#nbeats

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnBEATSModel(
    input_chunk_length=24,
    output_chunk_length=24,
    generic_architecture=True,
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    expansion_coefficient_dim=5,
    trend_polynomial_degree=2,
    dropout=0.0,
    activation="ReLU",
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)

#nhtis

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnHiTSModel(
    input_chunk_length=24,
    output_chunk_length=24,
    num_stacks=3,
    num_blocks=1,
    num_layers=2,
    layer_widths=512,
    pooling_kernel_sizes=None,
    n_freq_downsample=None,
    dropout=0.1,
    activation="ReLU",
    MaxPool1d=True,
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)

#chronos

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnHiTSModel(
    input_chunk_length=24,
    output_chunk_length=24,
    num_stacks=3,
    num_blocks=1,
    num_layers=2,
    layer_widths=512,
    pooling_kernel_sizes=None,
    n_freq_downsample=None,
    dropout=0.1,
    activation="ReLU",
    MaxPool1d=True,
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)


#block rnn

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnBlockRNNModel(
    input_chunk_length=24,
    output_chunk_length=24,
    model="RNN",
    hidden_dim=25,
    n_rnn_layers=1,
    hidden_fc_sizes=None,
    dropout=0.0,
    activation="ReLU",
    use_static_covariates=True,
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)


# dlinear

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnDLinearModel(
    input_chunk_length=24,
    output_chunk_length=24,
    shared_weights=False,
    kernel_size=25,
    const_init=True,
    use_static_covariates=True,
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)


#nlinear

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

model = EnNLinearModel(
    input_chunk_length=24,
    output_chunk_length=24,
    shared_weights=False,
    const_init=True,
    normalize=True,
    use_static_covariates=True,
    num_samples=20,
    noise_std=1.0,
    noise_type="gaussian",
    optimizer_kwargs={"lr": 0.0003992005679645662},
    random_state=SEED,
    batch_size=64,
    n_epochs=1,
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)

