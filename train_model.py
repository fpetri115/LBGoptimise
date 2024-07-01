import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
import tensorflow_decision_forests as tfdf
import noise
import tools

def load_training_data():

    print("Loading Training Data (1/2)")
    sps_parameters = np.load("training_data/sps_parameters_150600000.npy")

    print("Loading Training Data (2/2)")
    photometry = np.load("training_data/photometry_150600000.npy")

    return (sps_parameters, photometry)

class df_model:

    def __init__(self, depth = 32):
        print("Initialising...")
        self.data = load_training_data()


        self._depth = 32
        self._trained_model = None
        print("Initialised")

    def load_model(self, name):
        self._trained_model = tf.keras.models.load_model("saved_models/"+name)
        
    def train_model(self, nsamples):

        sps_parameters, photometry = self.data

        #slice data
        full_params = sps_parameters[:nsamples, :]
        full_data = photometry[:nsamples, :]
        full_redshifts = full_params[:, 0]

        #apply noise
        noisy_stuff = noise.get_noisy_magnitudes(full_params, full_data, random_state=42, return_params=False)
        full_data = noisy_stuff[0][1]
        full_params = noisy_stuff[0][0]
        full_redshifts = full_params[:, 0]

        #rmag cut
        rmin = 20 #what should this be?
        rmax = 25 #what should this be?

        #apply cut
        selected_indexes = np.where((full_data[:, 2] > rmin) & (full_data[:, 2] < rmax) )[0]
        params = full_params[selected_indexes, :]
        data = full_data[selected_indexes, :]
        redshifts = full_redshifts[selected_indexes]

        #label training data
        labels = np.zeros_like(redshifts)
        z_min = 2.5
        z_max = 3.5
        u_indexes = np.where((redshifts > z_min) & (redshifts < z_max) )[0]#& (data[:, 0] - data[:, 1] > 0.0))[0]
        u_params = params[u_indexes, :]
        u_data = data[u_indexes, :]
        labels[u_indexes] = 1
        #plt.hist(u_params[:, 0])

        #From photometry want to predict label
        print(data.shape)
        data = tools.calculate_colours(data)

        #Setup training and validation data
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        validation_split = 0.9
        train_size = int(validation_split * data.shape[0])
        test_size = data.shape[0] - train_size

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)

        train_dataset = train_dataset.batch(batch_size=1000)

        #train model
        model = tfdf.keras.RandomForestModel(max_depth=self._depth, verbose=2)
        model.fit(train_dataset)
        
        #save model
        self._trained_model = model
        model.save("saved_models/test_"+str(self._depth)+"_"+str(data.shape[0]))

        model.compile(metrics=["accuracy"])
        evaluation = model.evaluate(test_dataset.batch(batch_size=1000), return_dict=True)

        for name, value in evaluation.items():
            print(f"{name}: {value:.4f}")

        predicted_labels = model.predict(test_dataset.batch(batch_size=1000))

        confidence_level = 0.2 #hyperparameter(ideal value depends on survey size? The larger your initial set of galaxies, the lower your efficency can be)
        predictions = predicted_labels[:, 0]
        predicted_labels_at_confidence = np.where(predictions > confidence_level, 1.0, 0.0)
        cmatrix = tf.math.confusion_matrix(labels[train_size:], predicted_labels_at_confidence)

        print(cmatrix)
        # TN FP
        # FN TP

        z_bins = np.linspace(0.0, 7.0, 70)

        test_set_redshifts = redshifts[train_size:]
        test_set_lbgs = test_set_redshifts[np.where((test_set_redshifts > 2.5) & (test_set_redshifts < 3.5) )[0]]

        selected_lbgs = test_set_redshifts[np.where(predicted_labels_at_confidence == 1.0)[0]]

        plt.hist(test_set_lbgs, bins=z_bins, label="True LBGs", density=True, alpha=0.5)
        plt.hist(selected_lbgs, bins=z_bins, label="Selected LBGs", density=True, alpha=0.5)
        plt.legend()

        ninterlopers =  (np.where(selected_lbgs < 1.5)[0]).shape[0]

        print("efficency(%): ", float(((cmatrix[1, 1])/test_set_lbgs.shape[0])*100))
        print("purity(%): ", float((cmatrix[1, 1]/(cmatrix[0,1]+cmatrix[1, 1]))*100)) #of all positives - which are truly lbgs?
        print("interloper fraction(%): ", (ninterlopers/selected_lbgs.shape[0])*100) 


        #data_test = data[train_size:]
        #umg = data_test[:, 0]
        #gmr = data_test[:, 1]
        #selected_lbgs_trad = test_set_redshifts[np.where((gmr > -1.0) & (gmr < 1.2) & (umg > 0.75 + 1.5*gmr))[0]]
        #plt.hist(test_set_lbgs, bins=z_bins, label="True LBGs", density=True, alpha=0.5)
        #plt.hist(selected_lbgs, bins=z_bins, label="Selected LBGs", density=True, alpha=0.5)
        #plt.hist(selected_lbgs_trad, bins=z_bins, label="CARS LBG Cut", density=True, alpha=0.5)
        #plt.legend()

        model = keras.models.load_model("saved_models/test_32_3642")
        model.compile()
        evaluation = model.evaluate(test_dataset.batch(batch_size=1000), return_dict=True)
        predicted_labels = model.predict(test_dataset.batch(batch_size=1000))
        print(predicted_labels, flush=True)