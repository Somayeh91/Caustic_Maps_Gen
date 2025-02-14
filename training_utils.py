import random
import os
import pickle as pkl
from maps_util import read_all_maps, split_data
from my_models import *
from maps_util import NormalizeData, read_all_lens_pos, read_binary_map
from more_info import best_AD_models_info
from maps_util import read_AD_model
from metric_utils import calculate_ks_metric, process_FID
from FID_calculator import read_inceptionV3
from prepare_maps import process_lc4
from plotting_utils import compareinout


def display_model(model):
    model.summary()



def scheduler(epoch, lr):
    """Decrease learning rate by a factor of 0.5 every 20 epochs"""
    if epoch % 20 == 0 and epoch != 0:
        return lr * 0.5
    return lr


def compile_model(model,
                  learning_rate,
                  optimizer,
                  loss):
    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss=loss)
    return model


def fit_model(model_design_key,
              model,
              epochs,
              x_train,
              y_train=None,
              x_validation=None,
              y_validation=None,
              filepath=None,
              batch_size=8,
              early_callback_=None,
              use_multiprocessing=True):
    """
        Trains a machine learning model with optional early stopping, model checkpointing, learning rate scheduling,
        or output visualization.

        Parameters:
        -----------
        model_design_key : str
            A string key indicating the model design type, which influences dataset formatting and callback behavior.
        model : keras.Model
            The neural network model to be trained.
        epochs : int
            Number of training epochs.
        x_train : array-like or TensorFlow dataset
            The training data.
        y_train : array-like, optional
            The training labels, required for specific model types.
        x_validation : array-like or TensorFlow dataset, optional
            The validation data. If 'plot_output' callback is used, this must be provided.
        y_validation : array-like, optional
            The validation labels, used when `x_validation` is provided.
        filepath : str, optional
            Path to save the model checkpoint when `early_callback_` is set to 'model_checkpoint' or
            'plot_output' callback is used.
        batch_size : int, default=8
            Batch size for training.
        early_callback_ : str, optional
            Specifies an early stopping mechanism:
            - 'early_stop': Stops training if validation loss does not improve.
            - 'model_checkpoint': Saves model checkpoints at each epoch.
            - 'changing_lr': Adjusts the learning rate dynamically using a scheduler.
            - 'plot_output': Generates visual comparisons of model predictions at regular intervals (requires `x_validation`).
        use_multiprocessing : bool, default=True
            Whether to use multiprocessing during training.

        Returns:
        --------
        history : keras.callbacks.History
            A history object containing details of the training process.

        Notes:
        ------
        - If `model_design_key` contains 'lens', the `'plot_output'` callback enables lens position tracking.
        - If `model_design_key` starts with 'kgs' or 'bt', training is done with a model other than an autoencoder.
        - Otherwise, `x_train` and `x_validation` are converted to TensorFlow datasets to enable pre-fetching.
        """
    ec = []
    if early_callback_ is not None:
        for callback in early_callback_:
            ec = set_callbacks(callback, model_design_key, model, x_validation, filepath, ec)

    if not (model_design_key.startswith('kgs') or model_design_key.startswith('bt')):
        x_train = x_train.to_tf_dataset()
        x_validation = x_validation.to_tf_dataset()


        history = model.fit(x_train,
                            validation_data=x_validation,
                            epochs=epochs,
                            callbacks=ec,
                            use_multiprocessing=use_multiprocessing)

    else:
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=epochs,
                            validation_split=0.2,
                            callbacks=ec)
    return history


def prepare_input_to_fit_keras_ADs(running_params, model_params):
    """
    Prepares input data for training and testing a Keras-based Autoencoder model.

    Parameters:
    -----------
    running_params : dict
        Dictionary containing runtime parameters, including:
        - 'sample_size': Number of samples to be used.
        - 'mode': Defines the execution mode ('train_test', 'retrain_test', or 'test').
        - 'test_selection': Strategy for selecting test samples (e.g., 'random', 'all_test', 'sorted', etc.).
        - 'train_selection': Strategy for selecting training samples (e.g., 'k=g', 'random', 'retrain_old', etc.).
        - 'output_dir': Directory to save partitioned sample sets.
        - 'n_test_set': Number of test samples to select.
        - 'list_IDs_directory': Path to file containing map IDs for training, if applicable.
        - 'saved_model_path': Path to a previously saved model, used in retraining/testing scenarios.
        - 'batch_size': Batch size for data partitioning.

    model_params : dict
        Dictionary containing model parameters, used for retraining or testing.

    Returns:
    --------
    partition : dict
        Dictionary containing the partitioned datasets:
        - 'train': Training set map IDs.
        - 'validation': Validation set map IDs.
        - 'test': Test set map IDs.
    data_dict : dict or None
        Dictionary containing all map data if loaded, otherwise None.
    running_params : dict
        Updated running parameters in case of retraining or loading previous settings.
    model_params : dict
        Updated model parameters if reloaded from a saved model.

    Notes:
    ------
    - If `mode` is 'train_test', new training, validation, and test partitions are created.
    - If `mode` is 'retrain_test' or 'test', partitions and parameters may be loaded from a saved model.
    - Different `train_selection` strategies determine how training data is chosen (e.g., loading from a file vs. random sampling).
    - The function saves the generated partition to a pickle file in the `output_dir`.
    - The partitioning of data is done using `split_data()`, which divides the dataset into training, validation, and test sets.
    - A seeded random shuffling mechanism ensures reproducibility.
    """
    sample_size = running_params['sample_size']
    mode = running_params['mode']
    test_selection = running_params['test_selection']
    train_selection = running_params['train_selection']
    output_dir = running_params['output_dir']
    n_test_set = running_params['n_test_set']

    if mode == 'train_test':

        if train_selection == 'k=g' or train_selection == 'repeated_kg':
            ls_maps = np.loadtxt(running_params['list_IDs_directory'], dtype=int)
            data_dict = None

        else:
            maps_per_batch = 1122
            n_batches = int(((sample_size / maps_per_batch) - (sample_size / maps_per_batch) % 1))
            data_dict = read_all_maps(n_batches)
            all_keys = data_dict.keys()
            ls_maps = random.sample(list(all_keys), int(sample_size))

        partition = {}

        random.seed(10)
        shuffler = np.random.permutation(len(ls_maps))

        shuffler = random.sample(list(shuffler), int(sample_size))
        ls_maps = ls_maps[shuffler]
        n_maps = len(ls_maps)

        indx1, indx2, indx3 = split_data(n_maps, running_params['batch_size'], test_selection, n_test_set)

        partition['train'] = ls_maps[indx1]
        partition['validation'] = ls_maps[indx2]
        all_test_set = ls_maps[indx3]

        if test_selection == 'random':
            partition['test'] = np.asarray(random.sample(list(all_test_set), n_test_set))
        elif test_selection == 'all_test':
            partition['test'] = all_test_set
        elif test_selection == 'all_train':
            partition['test'] = partition['train']
        elif test_selection == 'all_data':
            partition['test'] = ls_maps
        elif test_selection == 'sorted':
            partition['test'] = np.sort(all_test_set)[:n_test_set]
        elif test_selection == 'given':
            partition['test'] = np.loadtxt(running_params['test_IDs'])

        f = open(output_dir + 'sample_set_indexes.pkl', 'wb')
        pkl.dump(partition, f)

    elif mode == 'retrain_test' or mode == 'test':
        if train_selection == 'retrain_old':
            partition_direc = running_params['saved_model_path'].split("model")[0] + 'sample_set_indexes.pkl'
            file_partition = open(partition_direc, 'rb')
            partition = pkl.load(file_partition)

            file_params = open(running_params['saved_model_path'].split('model')[0] + 'params.pkl', 'rb')
            new_running_params = pkl.load(file_params)

            file_params = open(running_params['saved_model_path'].split('model')[0] + 'model_params.pkl', 'rb')
            model_params = pkl.load(file_params)
            all_test_set = partition['test']
            ls_maps = list(partition['train']) + list(partition['validation']) + list(partition['test'])
            data_dict = None
            running_params = new_running_params

        else:
            if train_selection == 'random' or train_selection == 'retrain_random':
                maps_per_batch = 1122
                n_batches = int(((sample_size / maps_per_batch) - (sample_size / maps_per_batch) % 1))
                data_dict = read_all_maps(n_batches)
                all_keys = data_dict.keys()
                ls_maps = random.sample(list(all_keys), int(sample_size))

            elif train_selection == 'retrain_k=g' or\
                 train_selection == 'retrain_repeated_kg' or\
                 train_selection == 'repeated_kg':
                ls_maps = np.loadtxt(running_params['list_IDs_directory'], dtype=int)
                data_dict = None

            partition = {}

            random.seed(10)
            shuffler = np.random.permutation(len(ls_maps))

            shuffler = random.sample(list(shuffler), int(sample_size))
            ls_maps = ls_maps[shuffler]
            n_maps = len(ls_maps)

            indx1, indx2, indx3 = split_data(n_maps, running_params['batch_size'], test_selection, n_test_set)

            partition['train'] = ls_maps[indx1]
            partition['validation'] = ls_maps[indx2]
            all_test_set = ls_maps[indx3]

        if test_selection == 'random':
            partition['test'] = np.asarray(random.sample(list(all_test_set), n_test_set))
        elif test_selection == 'all_test':
            partition['test'] = all_test_set
        elif test_selection == 'all_train':
            partition['test'] = partition['train']
        elif test_selection == 'all_data':
            partition['test'] = ls_maps
        elif test_selection == 'sorted':
            partition['test'] = np.sort(all_test_set)[:n_test_set]
        elif test_selection == 'given':
            partition['test'] = np.loadtxt(running_params['test_IDs'])

    print('Train set size=%i, Validation set size=%i, Test set size=%i. ' % (len(partition['train']),
                                                                             len(partition['validation']),
                                                                             len(partition['test'])))

    return partition, data_dict, running_params, model_params


def prepare_input_for_kgs_bt(model_design_key, running_params):
    """
    Prepares input data for training and testing a Keras-based model, specifically for models that take the kappa-gamma-shear (kgs)
    and produce bottleneck (bt) representations or vice versa.

    Parameters:
    -----------
    model_design_key : str
        Specifies the model type and determines if lens position data should be included
        (e.g., 'kgs_lens_pos_to_bt' includes lens positions).
    running_params : dict
        Dictionary containing runtime parameters, including:
        - 'test_selection': Strategy for selecting test samples (e.g., 'random', 'all_test', 'sorted', etc.).
        - 'train_selection': Strategy for selecting training samples (e.g., 'k=g', 'retrain_k=g', 'random', etc.).
        - 'n_test_set': Number of test samples to select.
        - 'saved_LSR_path': Path to saved latent space representations (LSR).
        - 'test_IDs': List of specific test IDs, used if `test_selection` is 'given'.

    Returns:
    --------
    bt_train : numpy.ndarray
        Training set bottleneck representations.
    bt_test : numpy.ndarray
        Test set bottleneck representations.
    kgs_train : numpy.ndarray
        Normalized training set kappa, gamma, and shear parameters.
    kgs_test : numpy.ndarray
        Normalized test set kappa, gamma, and shear parameters.
    lens_pos_train : numpy.ndarray
        Training set lens position histograms (if applicable).
    lens_pos_test : numpy.ndarray
        Test set lens position histograms (if applicable).
    ids_train : numpy.ndarray
        Training set IDs.
    ids_test : numpy.ndarray
        Test set IDs.
    IDs : numpy.ndarray
        Full list of dataset IDs.

    Notes:
    ------
    - Reads metadata (`all_maps_meta_kgs.csv`) and loads latent space representations (`saved_LSR_path`).
    - If `model_design_key` is 'kgs_lens_pos_to_bt', lens position histograms are computed and normalized.
    - If `train_selection` is 'k=g' or 'retrain_k=g', the dataset is filtered to kappa ~ gamma within a given offset.
    - The dataset is expanded by a factor of 10 through augmentation, and the bottleneck representations
      are reshaped for input into the neural network.
    - The data is shuffled and split into 90% training and 10% test sets.
    - Training parameters are normalized using `NormalizeData()` before returning.
    - Different `test_selection` strategies determine how the test set is chosen.
    """
    test_selection = running_params['test_selection']
    train_selection = running_params['train_selection']
    n_test_set = running_params['n_test_set']
    print('Reading in data')
    all_params_ = pd.read_csv('./../data/all_maps_meta_kgs.csv')
    bottleneck_ = np.load(running_params['saved_LSR_path'])
    IDs_ = np.loadtxt('./../data/GD1_ids_list.txt', dtype=int)

    # An empty array to save the 2D histograms if needed, if not, it will remain zero
    lens_pos_ = np.zeros((len(IDs_), 20, 20, 1))

    all_params = all_params_[['ID', 'k', 'g', 's']].values
    x_params_ = np.asarray([all_params[:, 1:][all_params[:, 0] == ID][0] for ID in IDs_])

    if model_design_key == 'kgs_lens_pos_to_bt':
        all_lens_pos = read_all_lens_pos()

        # Looping through all LSRs
        for i, key in enumerate(IDs_):

            if all_lens_pos[key].shape[0] == 0:
                pass
            elif all_lens_pos[key].shape[0] != 0 and len(all_lens_pos[key].shape) == 1:
                lens_pos_[i, :, :, 0] = np.histogram2d(np.asarray([all_lens_pos[key]])[:, 0],
                                                       np.asarray([all_lens_pos[key]])[:, 1],
                                                       bins=20)[0]
            else:
                lens_pos_[i, :, :, 0] = np.histogram2d(all_lens_pos[key][:, 0],
                                                       all_lens_pos[key][:, 1],
                                                       bins=20)[0]

        print('Max of all lens pos 2D hists: ', np.max(lens_pos_))
        lens_pos_ = lens_pos_ / np.max(lens_pos_)

    if train_selection == 'k=g' or train_selection == 'retrain_k=g':
        offset = 0.3
        kappa = x_params_[:, 0]
        gamma = x_params_[:, 1]
        # Here we limit the paramater space to where gamma-offset<kappa<gamma+offset
        indx = np.where((kappa < gamma + offset) & (x_params_[:, 0] > gamma - offset))

        bottleneck = bottleneck_[indx]
        IDs = IDs_[indx]
        x_params = x_params_[indx]
        if model_design_key == 'kgs_lens_pos_to_bt':
            lens_pos = lens_pos_[indx]
    else:
        bottleneck = bottleneck_
        IDs = IDs_
        x_params = x_params_
        if model_design_key == 'kgs_lens_pos_to_bt':
            lens_pos = lens_pos_

    print('Normalizing the bottleneck by the maximum = %.5f' % np.max(bottleneck))
    bottleneck = bottleneck / np.max(bottleneck)
    n_lc = len(x_params)
    bottleneck_10 = np.zeros((n_lc * 10, 50, 50, 1))
    x_params_10 = np.zeros((n_lc * 10, 3))
    lens_pos_10 = np.zeros((n_lc * 10, 20, 20, 1))
    IDs_10 = np.zeros((n_lc * 10, 1))

    for i in range(n_lc):
        # print(i)
        for k in range(10):
            # print(k,i+i*k)
            bottleneck_10[10 * i + k] = bottleneck[i].reshape((50, 50, 1))
            x_params_10[10 * i + k] = x_params[i]
            lens_pos_10[10 * i + k] = lens_pos[i]
            IDs_10[10 * i + k] = IDs[i]

    n = len(x_params_10)
    train_n = int(0.9 * n)

    shuffler = np.random.permutation(n)
    bottleneck_10 = bottleneck_10[shuffler]
    x_params_10 = x_params_10[shuffler]
    IDs_10 = IDs_10[shuffler]
    y_train_ = x_params_10[:train_n]
    bt_train = bottleneck_10[:train_n]
    y_test_ = x_params_10[train_n:]
    bt_test_ = bottleneck_10[train_n:]

    if model_design_key == 'kgs_lens_pos_to_bt':
        lens_pos_10 = lens_pos_10[shuffler]
        lens_pos_train = lens_pos_10[:train_n]
        lens_pos_test_ = lens_pos_10[train_n:]
    else:
        lens_pos_train = np.zeros((len(bt_train)))
        lens_pos_test_ = np.zeros((len(bt_test_)))

    ids_train = IDs_10[:train_n]
    ids_test_ = IDs_10[train_n:]
    kgs_train = np.zeros(y_train_.shape)
    kgs_test_ = np.zeros(y_test_.shape)

    kgs_train[:, 0] = NormalizeData(y_train_[:, 0], np.max(y_train_[:, 0]), np.min(y_train_[:, 0]))
    kgs_train[:, 1] = NormalizeData(y_train_[:, 1], np.max(y_train_[:, 1]), np.min(y_train_[:, 1]))
    kgs_train[:, 2] = y_train_[:, 2]
    kgs_test_[:, 0] = NormalizeData(y_test_[:, 0], np.max(y_test_[:, 0]), np.min(y_test_[:, 0]))
    kgs_test_[:, 1] = NormalizeData(y_test_[:, 1], np.max(y_test_[:, 1]), np.min(y_test_[:, 1]))
    kgs_test_[:, 2] = y_test_[:, 2]

    if test_selection == 'random':
        kgs_test = kgs_test_
        bt_test = bt_test_
        lens_pos_test = lens_pos_test_
        ids_test = ids_test_
    elif test_selection == 'all_test':
        kgs_test = kgs_test_
        bt_test = bt_test_
        lens_pos_test = lens_pos_test_
        ids_test = ids_test_
    elif test_selection == 'all_train':
        kgs_test = kgs_train
        bt_test = bt_train
        lens_pos_test = lens_pos_train
        ids_test = ids_train
    elif test_selection == 'all_data':
        kgs_test = np.concatenate((kgs_test_, kgs_train), axis=0)
        bt_test = np.concatenate((bt_test_, bt_train), axis=0)
        lens_pos_test = np.concatenate((lens_pos_test_, lens_pos_train), axis=0)
        ids_test = np.concatenate((ids_test_, ids_train), axis=0)
    elif test_selection == 'sorted':
        kgs_test = kgs_test_[:n_test_set]
        bt_test = bt_test_[:n_test_set]
        lens_pos_test = lens_pos_test_[:n_test_set]
        ids_test = ids_test_[:n_test_set]
    elif test_selection == 'given':
        indexes = [True if id in running_params['test_IDs'] else False for id in IDs]
        # [np.where(IDs == id) for id in running_params['test_IDs']]
        bt_test = bottleneck[indexes]
        lens_pos_test = lens_pos[indexes]
        kgs_test = x_params[indexes]
        kgs_test[:, 0] = NormalizeData(kgs_test[:, 0], np.max(kgs_test[:, 0]), np.min(kgs_test[:, 0]))
        kgs_test[:, 1] = NormalizeData(kgs_test[:, 1], np.max(kgs_test[:, 1]), np.min(y_test_[:, 1]))
        ids_test = ids_test_[indexes]

    print('Train set size=%i, Validation set size=%i, Test set size=%i. ' % (len(bt_train),
                                                                             int(0.2 * len(bt_train)),
                                                                             len(bt_test)))
    return bt_train, bt_test, kgs_train, kgs_test, lens_pos_train, lens_pos_test, ids_train, ids_test, IDs


def set_up_model(model_design_key, model_params, running_params):
    """
    Initializes and configures a machine learning model based on the specified model design keys and parameters.

    Parameters:
    -----------
    model_design_key : str
        Specifies the type of model to initialize. Supported options include:
        - 'VAE_Unet_Resnet': Variational Autoencoder (VAE) with U-Net and ResNet components.
        - 'VAE_lens_pos': VAE with lens position encoding.
        - 'kgs_to_bt', 'bt_to_kgs', 'kgs_lens_pos_to_bt': Models for transforming kappa-gamma-shear (kgs) to bottleneck (bt) and vice versa.
        - Other VAE-based and convolutional models.
    model_params : dict
        Dictionary containing model parameters, including:
        - 'model_function': The function defining the model architecture.
        - 'z_size': Latent space dimension (for VAE models).
        - 'n_channels': Number of input/output channels.
        - 'input_side': Input size for some models.
        - 'input_side2': Secondary input shape (if applicable).
        - 'flow_label': Type of flow model (if applicable).
        - 'n_flows': Number of flows in flow-based architectures.
        - 'first_down_sampling': Downsampling factor for the first layer.
        - 'crop_scale': Scaling factor for lowering the pixel sizes of input images.
    running_params : dict
        Dictionary containing runtime parameters, including:
        - 'input_size': Original input size.
        - 'output_size': Processed output size.
        - 'res_scale': Resolution scaling factor.

    Returns:
    --------
    model : keras.Model
        The configured Keras model.

    Notes:
    ------
    - The function calculates the input dimension (`dim_input`) based on the provided resolution and cropping parameters.
    - If `model_design_key` starts with 'VAE', different encoder-decoder architectures are used based on the specific key.
    - If `crop_scale` is not 1.0 and `model_design_key` is 'Unet2', a U-Net model with downsampling is set up.
    - Certain models (e.g., `kgs_to_bt`, `bt_to_kgs`, `kgs_lens_pos_to_bt`) use specific functions defined in `model_params`.
    - The function dynamically selects the appropriate model function based on the given parameters.
    """
    dim = running_params['input_size']
    crop_scale = model_params['crop_scale']
    res_scale = running_params['res_scale']
    if dim == 10000:
        dim_input = int((dim / res_scale) * crop_scale)
    else:
        dim_input = running_params['output_size']

    if model_design_key.startswith('VAE'):
        if model_design_key == 'VAE_Unet_Resnet':
            encoder = vae_encoder(dim_input,
                                  z_size=model_params['z_size'],
                                  n_channels=model_params['n_channels'],
                                  af='relu')
            model = VAE(encoder,
                        vae_decoder(model_params['z_size'], 'relu'))
        elif model_design_key == 'VAE_lens_pos':
            encoder = vae_encoder_lens_pos()
            decoder = vae_decoder_lens_pos()
            model = VAE2(encoder,
                         vae_decoder(z_size=625))
        else:
            encoder = vae_encoder_3params(dim_input,
                                          input2=3,
                                          z_size=model_params['z_size'],
                                          n_channels=model_params['n_channels'],
                                          af='relu')
            model = VAE(encoder,
                        vae_decoder(model_params['z_size'], 'relu'))

    elif crop_scale != 1. and model_design_key == 'Unet2':
        model = model_params['model_function'](dim_input, first_down_sampling=4)

    elif model_design_key == 'kgs_to_bt' or \
            model_design_key == 'bt_to_kgs' or \
            model_design_key == 'kgs_lens_pos_to_bt':

        model = model_params['model_function'](model_params['input_side'],
                                               input2_shape=model_params['input_side2'])

    else:
        model = model_params['model_function'](dim_input,
                                               input2=model_params['input_side2'],
                                               n_channels=model_params['n_channels'],
                                               z_size=model_params['z_size'],
                                               flow_label=model_params['flow_label'],
                                               n_flows=model_params['n_flows'],
                                               first_down_sampling=model_params['first_down_sampling'],
                                               af='relu')

    return model


def evaluate_kgs_generated_maps(IDs, AD_model_ID, LSR_generated, running_params, metric):
    """
    Evaluates the quality of kappa-gamma-shear (kgs) generated maps using a specified metric.

    Parameters:
    -----------
    IDs : list or numpy.ndarray
        List of map IDs corresponding to the generated maps.
    AD_model_ID : str
        Identifier for the anomaly detection (AD) model used for evaluation.
    LSR_generated : numpy.ndarray
        Latent space representations (LSR) of the generated maps.
    running_params : dict
        Dictionary containing runtime parameters, including:
        - 'saved_LSR_path': Path where latent space representations are stored.
        - 'input_size': Expected input size of the maps.
        - 'output_dir': Directory to save metric results.
    metric : str
        The evaluation metric to use. Options include:
        - 'ssm': Uses the Anderson-Darling test to compute the Kolmogorov-Smirnov metric.
        - 'lc_sim': Processes light curve similarity using a predefined function.
        - 'fid': Computes the Fréchet Inception Distance (FID) score.

    Returns:
    --------
    float or numpy.ndarray
        - If `metric` is 'fid', returns the computed FID score.
        - Otherwise, returns the median value of the computed metric across all maps.

    Notes:
    ------
    - Loads the best-performing anomaly detection (AD) model based on `AD_model_ID` and reconstructs the maps.
    - Uses a predefined decoder layer to reconstruct maps from the latent space representations.
    - Reads the ground truth maps, normalizes them, and compares them against the reconstructed maps using the chosen metric.
    - For 'fid' evaluation, loads the InceptionV3 model and computes the FID score.
    - Saves metric-related outputs in `output_dir`.
    """
    model_file = \
    np.asarray(best_AD_models_info['job_model_filename'])[np.asarray(best_AD_models_info['job_names']) == AD_model_ID][
        0]
    cost_label = \
    np.asarray(best_AD_models_info['job_cost_labels'])[np.asarray(best_AD_models_info['job_names']) == AD_model_ID][0]

    path_to_model = running_params['saved_LSR_path'].split('LSR')[0]
    model = read_AD_model(AD_model_ID, model_file, cost_label)
    dim = running_params['input_size']
    output_dir = running_params['output_dir']

    if dim == 50:
        decoder_output = model.get_layer('conv2d_transpose_6').output
        decoder_input = model.get_layer('conv2d_transpose').input
    else:
        print('Unknown name of the decoder input and output layers.')
        return None
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)
    map_reconstruction = decoder.predict(LSR_generated)
    maps = np.zeros((len(map_reconstruction), 1000, 1000, 1))
    metric_all = np.zeros((len(IDs)))
    for i, ID in enumerate(IDs):
        map_i = NormalizeData(np.log10(read_binary_map(ID, scaling_factor=10, to_mag=True) + 0.004))
        maps[i] = map_i
        map_new = map_reconstruction[i]
        if metric == 'ssm':
            metric_all[i] = calculate_ks_metric(map_i, map_new, test_name='anderson')

        elif metric == 'lc_sim':
            metric_all[i] = process_lc4([model + '_' + str(ID),
                                         map_i,
                                         map_new,
                                         None,
                                         1000,
                                         1000,
                                         'AD',
                                         'random',
                                         output_dir + '%s-metric_for_ID_%i' % (metric, ID),
                                         False])
    if metric == 'fid':
        inceptionv3 = read_inceptionV3()
        return process_FID([maps,
                            map_reconstruction,
                            inceptionv3])
    else:
        return np.median(metric_all)


def evaluate_AD_maps(in_maps, out_maps, model_name):
    """
    Evaluates the quality of anomaly detection (AD) maps by computing various similarity metrics.

    Parameters:
    -----------
    in_maps : numpy.ndarray
        Array of original input maps.
    out_maps : numpy.ndarray
        Array of reconstructed or generated output maps to be compared against `in_maps`.
    model_name : str
        Name of the model used for generating `out_maps`, included in light curve similarity computation.

    Returns:
    --------
    metric_all : numpy.ndarray
        A NumPy array of shape (N, 3), where N is the number of maps, and each row contains:
        - Column 0: The Kolmogorov-Smirnov metric using the Anderson-Darling test ('ssm').
        - Column 1: The mean light curve similarity score ('lc_sim').
        - Column 2: The Fréchet Inception Distance (FID) score, which is the same for all maps.

    Notes:
    ------
    - The function computes three evaluation metrics for each map:
      1. The Anderson-Darling variant of the Kolmogorov-Smirnov metric (`calculate_ks_metric`).
      2. The light curve similarity (`process_lc4`), averaged over multiple runs.
      3. The Fréchet Inception Distance (FID) score, computed using a pre-trained InceptionV3 model.
    - FID is computed once for the entire dataset and assigned uniformly to all entries in `metric_all`.
    - The function assumes `map_i` and `map_new` have compatible shapes and are properly preprocessed.
    """
    metric_all = np.zeros((len(in_maps), 3))
    for i, ID in enumerate(in_maps):
        map_i = in_maps[i]
        map_new = out_maps[i]
        # Calculating Metric SSM
        metric_all[i, 0] = calculate_ks_metric(map_i.flatten(), map_new.flatten(), test_name='anderson')

        # Calculating metric lc_sim
        tmp_lc_sim = process_lc4([model_name,
                                  map_i,
                                  map_new.reshape(map_i.shape),
                                  None,
                                  100,
                                  100,
                                  'AD',
                                  'random',
                                  None,
                                  False])
        metric_all[i, 1] = np.mean(np.asarray(tmp_lc_sim).flatten())

    inceptionv3 = read_inceptionV3()
    metric_all[:, 2] = np.ones((len(in_maps))) * process_FID([in_maps,
                                                              out_maps,
                                                              inceptionv3])
    return metric_all


class ComparisonPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 model,
                 validation_data,
                 model_design_key,
                 lens_pos=False,
                 save_dir="comparison_images",
                 interval=10):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.interval = interval
        self.save_dir = save_dir
        self.lens_pos = lens_pos
        self.model_design_key = model_design_key
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:  # Every 10 epochs
            print(f"\nGenerating comparison plot at epoch {epoch + 1}...")

            # Get one sample batch
            X, y_true = next(iter(self.validation_data))

            # Pick the first sample in batch
            if self.lens_pos:
                (X1, X2) = X
                input_img = X1[0]
            else:
                input_img = X[0]
            true_output = y_true[0]


            # Model prediction
            predicted_output = self.model.predict(X, verbose=0)[0]

            # Plot results
            dim = input_img.shape[1]
            compareinout(predicted_output,
                         input_img,
                         dim,
                         self.save_dir,
                         self.model_design_key,
                         epoch)
            print(f"Saved comparison plot: {self.save_dir+ 'model_%s_test_%i.png' % (self.model_design_key, epoch)}")


def set_callbacks(callback_label, model_design_key, model, x_validation, filepath, ec):
    if callback_label == 'early_stop':
        ec.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.00001))
    elif callback_label == 'model_checkpoint':
        ec.append(ModelCheckpoint(
            filepath=filepath,
            save_freq='epoch'))
    elif callback_label == 'changing_lr':
        ec.append(keras.callbacks.LearningRateScheduler(scheduler))
    elif callback_label == 'plot_output':
        if not x_validation == 'None':
            if 'lens' in model_design_key.split('_'):
                lens_pos_flag = True
            else:
                lens_pos_flag = False
            ec.append(ComparisonPlotCallback(model,
                                             x_validation,
                                             model_design_key,
                                             lens_pos=lens_pos_flag,
                                             save_dir=filepath,
                                             interval=10))
    return ec