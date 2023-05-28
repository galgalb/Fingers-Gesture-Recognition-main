import os.path
import pickle
import numpy as np

from datetime import datetime

from Source.streamer.data import Data
from Source.streamer.viz import Viz
from Source.fgr.data_collection import Experiment
from Source.fgr import models
from Source.fgr.data_manager import Recording
from Source.fgr.pipelines import Data_Pipeline

# run the experiment until we get accuracy of 90% or more
# first run the experiment with 4 repetitions
# if the accuracy is less than 90%, then run the experiment with 2 repetitions each additional time
# each time add the data to the dataset and retrain the model
# after each run, save the model and the dataset
# stop even if the accuracy is 90% or more if the total number of repetitions is 10

best_accuracy = 0
exp_n_repetitions = [4, 2, 2, 2]
X_dataset = []
label_dataset = []

for j in range(len(exp_n_repetitions)):
    if best_accuracy >= 0.9 and i > 0:
        break

    """ RUN PARAMETERS """

    host_name = "127.0.0.1"  # IP address from which to receive data
    port = 20001  # Local port through which to access host
    n_repetitions = exp_n_repetitions(j)  # Number of repetitions of each gesture during data collection
    signal_check = False   # View real-time signals as signal quality check? Note: running this precludes PsychoPy (idk why) so run it once with True then run again with False

    """SIGNAL CHECK """

    if signal_check:
        timeout = None
        verbose = True
        data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)
        data_collector.start()
        plotter = Viz(data_collector, plot_imu=False, plot_ica=False, ylim_exg=(-350, 350), update_interval_ms=50,
                      max_timeout=20)
        plotter.start()
        exit(0)

    """ DATA COLLECTION """

    start_time = datetime.now()

    timeout = 15  # if streaming is interrupted for this many seconds or longer, terminate program
    verbose = False  # if to print to console BT packet summary (as sanity check) upon each received packet
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)

    # PsychoPy experiment
    data_dir = 'data'
    exp = Experiment()
    exp.run(data_collector, data_dir, n_repetitions=n_repetitions, fullscreen=False, img_secs=3, screen_num=0,exp_num=j)

    calibration_time = datetime.now()
    print('Data collection complete. Beginning model creation...')

    """ MODEL CREATION """

    # Get data and labels
    rec_obj = Recording([data_collector.save_as], Data_Pipeline(),j)  # noqa - the save_as param is updated in the experiment.run() method
    X, labels = rec_obj.get_dataset()
    # add the data to the dataset
    X_dataset.append(X)
    label_dataset.append(labels)

    # Train models with the full dataset
    n_models = min(5, n_repetitions)
    n_classes = len(os.listdir('images'))
    model = models.Net(num_classes=n_classes, dropout_rate=0.1)

    models, accuracies = model.cv_fit_model(X_dataset, label_dataset,
                                            num_epochs=200,
                                            batch_size=64,
                                            lr=0.001,
                                            l2_weight=0.0001,
                                            num_folds=n_models)

    # Evaluate models
    print(f'model average accuracy: {np.mean(accuracies)}')
    accs = []
    for i, model in enumerate(models):
        accs.append(model.evaluate_model(model.test_data, model.test_labels, cm_title='model number ' + str(i)))

    # Save best model with experiment number
    best_acc = np.argmax(accs)
    print(f'best model test accuracy: {accs[best_acc] * 100:0.2f}%')
    model_fp = rf'models\{os.path.basename(os.path.splitext(data_collector.save_as)[0])}_{n_classes}gestures_{accs[best_acc] * 100:0.2f}%_rep_{j}.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_fp, 'wb') as f:
        pickle.dump(models[best_acc], f)
        print(f'Saved model to {model_fp}')

    end_time = datetime.now()
    calibration_duration = calibration_time - start_time
    train_duration = end_time - calibration_time
    process_duration = end_time - start_time
    print('Training process complete.')
    print(f'Calibration took {calibration_duration}')
    print(f'Training took {train_duration}')
    print(f'Whole process took {process_duration}')