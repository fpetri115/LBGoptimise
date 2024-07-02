import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
import tensorflow_decision_forests as tfdf
import noise
import tools

def load_training_data(id):
    """get training data from files
    """
    
    print("Loading Training Data (1/2)")
    sps_parameters = np.load("training_data/sps_parameters_"+str(id)+".npy")

    print("Loading Training Data (2/2)")
    photometry = np.load("training_data/photometry_"+str(id)+".npy")

    return (sps_parameters, photometry)

def load_model(name):
    """load model given string filename
    """
    model = keras.models.load_model("saved_models/"+name)
    model.compile(metrics=["accuracy"])
    return model

def save_model(model, depth, name):
    """save model given string filename
    """
    model.save("saved_models/test_"+str(depth)+"_"+name)
    
def evaluate_model(model, dataset):
    """for a given tensorflow dataset, evaluate model to give
    accuracy"""
    evaluation = model.evaluate(dataset, return_dict=True)

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

def make_prediction(model, dataset):
    """return predicted labels (between 0 and 1) for
    a given tensorflow dataset
    """
    labels = model.predict(dataset)
    return labels[:, 0]

def make_prediction_at_confidence(model, dataset, confidence_level):
    """return predicted labels (0 or 1) for
    a given tensorflow dataset, for given a confidence level
    """
    labels = make_prediction(model, dataset)
    predicted_labels_at_confidence = np.where(labels > confidence_level, 1.0, 0.0)
    return predicted_labels_at_confidence

def prepare_data(sps_parameters, photometry):
    """take training data numpy arrays and convert into colours with
    labels (in redshift range or no). Also returns redshift
    """
    sps_parameters, photometry = _apply_noise_and_selection(sps_parameters, photometry)
    sps_parameters, photometry = _apply_r_mag_cutoffs(sps_parameters, photometry, 20, 25)
    redshifts = sps_parameters[:, 0]
    labels = _generate_training_data_labels(redshifts, 2.5, 3.5)

    data = tools.calculate_colours(photometry)

    return data, labels, redshifts

def create_test_validation_data(data, labels, validation_split, batch_size):
    """turns output of prepare data into tensorflow dataset, given test/training
    split and batch size
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    validation_split = validation_split
    train_size = int(validation_split * data.shape[0])
    test_size = data.shape[0] - train_size

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    return train_dataset, test_dataset, train_size, test_size

def train_model(sps_parameters, photometry, depth, split, batch_size):
    """returns trained model on given data"""
    data, labels, redshifts = prepare_data(sps_parameters, photometry)
    train_dataset, test_dataset, train_size, test_size = create_test_validation_data(data, labels, split, batch_size=batch_size)

    #train model
    model = tfdf.keras.RandomForestModel(max_depth=depth, verbose=2)
    model.fit(train_dataset)
    save_model(model, depth, str(data.shape[0]))

    model.compile(metrics=["accuracy"])
    evaluate_model(model, test_dataset)

    return model

def get_optimised_nz(model, dataset, redshifts, confidence_level):
    predicted_labels = make_prediction_at_confidence(model, dataset, confidence_level)
    selected = redshifts[np.where(predicted_labels == 1.0)[0]]
    return selected

def get_binned_nz(redshifts):
    binned = redshifts[np.where((redshifts > 2.5) & (redshifts < 3.5) )[0]]
    return binned

def get_interlopers(redshifts):
    interlopers = redshifts[np.where(redshifts < 1.5)[0]]
    return interlopers

def confusion_matrix(original_labels, predicted_labels):
    cmatrix = tf.math.confusion_matrix(original_labels, predicted_labels)
    print(cmatrix)
    # TN FP
    # FN TP
    return cmatrix

def evaluate_model_performance(model, dataset, redshifts, original_labels, confidence_level):

    cmatrix = confusion_matrix(original_labels, make_prediction_at_confidence(model, dataset, confidence_level))
    selected_objects = get_optimised_nz(model, dataset, redshifts, confidence_level)
    original_objects = get_binned_nz(redshifts)
    ninterlopers =  get_interlopers(selected_objects).shape[0]

    print("efficency(%): ", float(((cmatrix[1, 1])/original_objects.shape[0])*100))
    print("purity(%): ", float((cmatrix[1, 1]/(cmatrix[0,1]+cmatrix[1, 1]))*100)) #of all positives - which are truly lbgs?
    print("interloper fraction(%): ", (ninterlopers/selected_objects.shape[0])*100) 

def get_classic_udropouts(colours, redshifts):

    umg = colours[:, 0]
    gmr = colours[:, 1]
    selected_lbgs_classic = redshifts[np.where((gmr > -1.0) & (gmr < 1.2) & (umg > 0.75 + 1.5*gmr))[0]]
    return selected_lbgs_classic

def _get_subsample_of_data(sps_parameters, photometry, nsamples):
    #slice data
    sel_params = sps_parameters[:nsamples, :]
    sel_data = photometry[:nsamples, :]

    return sel_params, sel_data

def _apply_noise_and_selection(params, data):

    noisy_stuff = noise.get_noisy_magnitudes(params, data, random_state=42, return_params=False)
    sel_data = noisy_stuff[0][1]
    sel_params = noisy_stuff[0][0]

    return sel_params, sel_data

def _apply_r_mag_cutoffs(params, data, rmin, rmax):

    selected_indexes = np.where((data[:, 2] > rmin) & (data[:, 2] < rmax) )[0]
    params_cut = params[selected_indexes, :]
    data_cut = data[selected_indexes, :]

    return params_cut, data_cut

def _generate_training_data_labels(redshifts, zmin, zmax,):

    labels = np.zeros_like(redshifts)
    u_indexes = np.where((redshifts > zmin) & (redshifts < zmax) )[0]#& (data[:, 0] - data[:, 1] > 0.0))[0]
    labels[u_indexes] = 1

    return labels
    #u_params = params[u_indexes, :]
    #u_data = data[u_indexes, :]
    #plt.hist(u_params[:, 0])

    