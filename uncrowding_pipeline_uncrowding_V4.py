from stimulus_set_generator import StimulusMaker

from candidate_models.model_commitments import brain_translated_pool
from brainio.stimuli import StimulusSet
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from brainscore.metrics.accuracy import Accuracy
from scipy.optimize import least_squares
from scipy.stats import t
from joblib import parallel_backend
import time

t_obj = time.localtime()
current_time = time.strftime("%H:%M:%S", t_obj)


class Uncrowding:
    def __init__(self, modelname, region, time_bins, train_metadata=None, test_metadata=None, directory=None):
        self.model = brain_translated_pool[modelname]
        self.model_string = modelname
        self.region = region
        self.model.start_recording(region, time_bins)
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.directory = directory
        self.predictions = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.offsets_test = None
        self.noise_test = None

    def make_verniers_stimulus_set(self, batch_size, directory_name, mode='verniers', split=0.5):
        try:
            os.mkdir(directory_name)
            self.directory = directory_name
        except OSError as error:
            print(error)
            self.directory = directory_name

        if mode == 'uncrowding':
            n_shapes = 5
        elif mode == 'crowding':
            n_shapes = 1
        else:
            n_shapes = 5

        stimuli_dict_train = {
            'stimulus_id': [],
            'filename': [],
            'resolution': [],
            'offset': [],
            'value': [],
            'noise': []
        }

        stimuli_dict_test = {
            'stimulus_id': [],
            'filename': [],
            'resolution': [],
            'offset': [],
            'value': [],
            'noise': []
        }

        train_index = 0
        test_index = 0

        for noise in [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]:
            MyMaker = StimulusMaker(image_size=(224, 224), patch_size=40, bar_width=2)
            test_image_data = MyMaker.make_test_batch(selected_shape=1, n_shapes=[n_shapes],
                                                      batch_size=batch_size, stimulus_condition=2,
                                                      centralize=False, reduce_df=False)

            for i in range(0, int(np.floor(batch_size * split))):
                vernier = test_image_data[0][i]
                flankers = test_image_data[1][i]
                # whole_stimulus = vernier + flankers  # 3-dimensional array (X,Y,grayscale)

                value = "left" if test_image_data[3][i] else "right"  # extract binary value
                offset = test_image_data[4][i][0]

                if mode == 'verniers':
                    whole_stimulus = np.squeeze(vernier)
                else:
                    whole_stimulus = np.squeeze(vernier + flankers)  # squeeze to (X,Y)

                # noise
                whole_stimulus += np.abs(np.random.normal(0, noise, whole_stimulus.shape))

                ws = (((whole_stimulus - whole_stimulus.min()) / (whole_stimulus.max() - whole_stimulus.min())) * 255.9) \
                    .astype(np.uint8)
                im = Image.fromarray(ws)  # generate PIL.Image object

                filename = "test_" + str(test_index) + "_" + str(int(offset)) + "_" + value + "_" + str(
                    noise) + "_" + ".png"
                # print(filename)
                im.save(directory_name + "/" + filename)  # Save to directory
                stimuli_dict_test['stimulus_id'].append(str(test_index))  # Append metadata
                stimuli_dict_test['filename'].append(filename)
                stimuli_dict_test['resolution'].append(whole_stimulus.shape[0])
                stimuli_dict_test['offset'].append(offset)
                stimuli_dict_test['value'].append(value)
                stimuli_dict_test['noise'].append(noise)

                test_index += 1

            for i in range(int(np.floor(batch_size * split)), batch_size):
                vernier = test_image_data[0][i]
                flankers = test_image_data[1][i]
                # whole_stimulus = vernier + flankers  # 3-dimensional array (X,Y,grayscale)

                value = "left" if test_image_data[3][i] else "right"  # extract binary value
                offset = test_image_data[4][i][0]

                if mode == 'verniers':
                    whole_stimulus = np.squeeze(vernier)
                else:
                    whole_stimulus = np.squeeze(vernier + flankers)  # squeeze to (X,Y)

                # noise
                whole_stimulus += np.abs(np.random.normal(0, noise, whole_stimulus.shape))

                ws = (((whole_stimulus - whole_stimulus.min()) / (whole_stimulus.max() - whole_stimulus.min())) * 255.9) \
                    .astype(np.uint8)
                im = Image.fromarray(ws)  # generate PIL.Image object

                filename = "train_" + str(train_index) + "_" + str(int(offset)) + "_" + value + "_" + str(
                    noise) + "_" + ".png"
                im.save(directory_name + "/" + filename)  # Save to directory
                stimuli_dict_train['stimulus_id'].append(str(train_index))  # Append metadata
                stimuli_dict_train['filename'].append(filename)
                stimuli_dict_train['resolution'].append(whole_stimulus.shape[0])
                stimuli_dict_train['offset'].append(offset)
                stimuli_dict_train['value'].append(value)
                stimuli_dict_train['noise'].append(noise)

                train_index += 1

        df_train = pd.DataFrame(stimuli_dict_train)
        print(df_train)
        df_train.to_csv(self.directory + '_train' + '.csv')  # Export CSV

        df_test = pd.DataFrame(stimuli_dict_test)
        print(df_test)
        df_test.to_csv(self.directory + '_test' + '.csv')  # Export CSV
        return None

    def set_directory(self, directory_name):
        self.directory = directory_name

    def segmented_make_verniers_stimulus_set(self, batch_size, directory_name, mode='verniers', split=0.5):
        segment_size = 50
        try:
            os.mkdir(directory_name)
            self.directory = directory_name
        except OSError as error:
            print(error)
            self.directory = directory_name
        try:
            os.mkdir(directory_name + '_metadata')
        except OSError as error:
            print(error)

        if mode == 'uncrowding':
            n_shapes = 5
        elif mode == 'crowding':
            n_shapes = 1
        else:
            n_shapes = 5

        train_index = 0
        test_index = 0

        for noise in [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]:

            stimuli_dict_train = {
                'stimulus_id': [],
                'filename': [],
                'resolution': [],
                'offset': [],
                'value': [],
                'noise': []
            }

            stimuli_dict_test = {
                'stimulus_id': [],
                'filename': [],
                'resolution': [],
                'offset': [],
                'value': [],
                'noise': []
            }

            MyMaker = StimulusMaker(image_size=(224, 224), patch_size=40, bar_width=2)
            test_image_data = MyMaker.make_test_batch(selected_shape=1, n_shapes=[n_shapes],
                                                      batch_size=batch_size, stimulus_condition=2,
                                                      centralize=False, reduce_df=False)
            test_segment_index = 0
            for i in range(0, int(np.floor(batch_size * split))):
                if i % segment_size == 0 and i != 0:
                    test_segment_index += 1
                    df_test = pd.DataFrame(stimuli_dict_test)
                    df_test.to_csv(self.directory + '_metadata/' + 'test_' + str(noise) + '_' + str(
                        test_segment_index) + '.csv')  # Export CSV
                    stimuli_dict_test = {
                        'stimulus_id': [],
                        'filename': [],
                        'resolution': [],
                        'offset': [],
                        'value': [],
                        'noise': []
                    }

                vernier = test_image_data[0][i]
                flankers = test_image_data[1][i]
                # whole_stimulus = vernier + flankers  # 3-dimensional array (X,Y,grayscale)

                value = "left" if test_image_data[3][i] else "right"  # extract binary value
                offset = test_image_data[4][i][0]

                if mode == 'verniers':
                    whole_stimulus = np.squeeze(vernier)
                else:
                    whole_stimulus = np.squeeze(vernier + flankers)  # squeeze to (X,Y)

                # noise
                whole_stimulus += np.abs(np.random.normal(0, noise, whole_stimulus.shape))

                ws = (((whole_stimulus - whole_stimulus.min()) / (whole_stimulus.max() - whole_stimulus.min())) * 255.9) \
                    .astype(np.uint8)
                im = Image.fromarray(ws)  # generate PIL.Image object

                filename = "test_" + str(test_index) + "_" + str(int(offset)) + "_" + value + "_" + str(
                    noise) + "_" + ".png"
                im.save(directory_name + "/" + filename)  # Save to directory
                stimuli_dict_test['stimulus_id'].append(str(test_index))  # Append metadata
                stimuli_dict_test['filename'].append(filename)
                stimuli_dict_test['resolution'].append(whole_stimulus.shape[0])
                stimuli_dict_test['offset'].append(offset)
                stimuli_dict_test['value'].append(value)
                stimuli_dict_test['noise'].append(noise)

                test_index += 1

            test_segment_index += 1
            df_test = pd.DataFrame(stimuli_dict_test)
            df_test.to_csv(self.directory + '_metadata/' + 'test_' + str(noise) + '_' + str(
                test_segment_index) + '.csv')  # Export CSV

            train_segment_index = 0
            for i in range(int(np.floor(batch_size * split)), batch_size):

                if i % segment_size == 0 and i != int(np.floor(batch_size * split)):
                    train_segment_index += 1
                    df_train = pd.DataFrame(stimuli_dict_train)
                    df_train.to_csv(self.directory + '_metadata/' + 'train_' + str(noise) + '_' + str(
                        train_segment_index) + '.csv')  # Export CSV
                    stimuli_dict_train = {
                        'stimulus_id': [],
                        'filename': [],
                        'resolution': [],
                        'offset': [],
                        'value': [],
                        'noise': []
                    }

                vernier = test_image_data[0][i]
                flankers = test_image_data[1][i]
                # whole_stimulus = vernier + flankers  # 3-dimensional array (X,Y,grayscale)

                value = "left" if test_image_data[3][i] else "right"  # extract binary value
                offset = test_image_data[4][i][0]

                if mode == 'verniers':
                    whole_stimulus = np.squeeze(vernier)
                else:
                    whole_stimulus = np.squeeze(vernier + flankers)  # squeeze to (X,Y)

                # noise
                whole_stimulus += np.abs(np.random.normal(0, noise, whole_stimulus.shape))

                ws = (((whole_stimulus - whole_stimulus.min()) / (whole_stimulus.max() - whole_stimulus.min())) * 255.9) \
                    .astype(np.uint8)
                im = Image.fromarray(ws)  # generate PIL.Image object

                filename = "train_" + str(train_index) + "_" + str(int(offset)) + "_" + value + "_" + str(
                    noise) + "_" + ".png"
                im.save(directory_name + "/" + filename)  # Save to directory
                stimuli_dict_train['stimulus_id'].append(str(train_index))  # Append metadata
                stimuli_dict_train['filename'].append(filename)
                stimuli_dict_train['resolution'].append(whole_stimulus.shape[0])
                stimuli_dict_train['offset'].append(offset)
                stimuli_dict_train['value'].append(value)
                stimuli_dict_train['noise'].append(noise)

                train_index += 1

            train_segment_index += 1
            df_train = pd.DataFrame(stimuli_dict_train)
            df_train.to_csv(self.directory + '_metadata/' + 'train_' + str(noise) + '_' + str(
                train_segment_index) + '.csv')  # Export CSV

        return test_segment_index, train_segment_index

    def get_activations(self, split):
        if split == 'train':
            stimulus_set = StimulusSet.from_files(self.directory + '_train.csv', self.directory)
        elif split == 'test':
            stimulus_set = StimulusSet.from_files(self.directory + '_test.csv', self.directory)
        else:
            print('error: Incorrect split term used')
            return None
        activations = self.model.look_at(stimulus_set, number_of_trials=10) \
            .to_dataframe(name='activations').reset_index()
        return activations

    def segmented_get_activations_features(self, split, segmentations, mode):
        for noise_level in ['low', 'medium', 'high']:
            for i in range(1, segmentations):
                stimulus_set = StimulusSet. \
                    from_files(self.directory + '_metadata/' + split + '_' + noise_level + '_' + str(i) + '.csv',
                               self.directory)
                activations = self.model.look_at(stimulus_set, number_of_trials=10) \
                    .to_dataframe(name='activations').reset_index()
                seg_X, seg_y, seg_offsets, seg_noise = self.features(activations)
                with open(self.model_string + mode + self.region + 'times.txt', 'a') as f:
                    f.write(str(noise_level) + ': ' + split + ' segment ' + str(i) + " of " + str(segmentations) + str(
                        time.strftime("%H:%M:%S", time.localtime())) + '\n')
                if i == 1 and noise_level == 'low':
                    X = seg_X
                    y = seg_y
                    offsets = seg_offsets
                    noise = seg_noise
                else:
                    X = np.concatenate((X, seg_X))
                    y = np.concatenate((y, seg_y))
                    offsets = np.concatenate((offsets, seg_offsets))
                    noise = np.concatenate((noise, seg_noise))
        return X, y, offsets, noise

    @staticmethod
    def features(activations):
        print("Pivot...")
        X = activations.pivot(index='filename', columns='neuroid_num', values='activations').reset_index()
        print("Offsets and noise...")
        offsets = X['filename'].apply(lambda x: int(x.split('_')[2])).to_numpy()
        noise = X['filename'].apply(lambda x: x.split('_')[4]).to_numpy()
        print("Extracting features/labels...")
        y = X['filename'].apply(lambda x: 0 if 'left' in x else 1).to_numpy()
        X = X.iloc[:, 1:].to_numpy()
        return X, y, offsets, noise

    def fit_predict(self):
        train_activations = self.get_activations('train')
        test_activations = self.get_activations('test')
        self.X_train, self.y_train, _, _ = self.features(train_activations)
        self.X_test, self.y_test, self.offsets_test, self.noise_test = self.features(test_activations)

        # Fit to data
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=10, solver='liblinear'))
        print("Fitting model... ")
        clf.fit(self.X_train, self.y_train)

        # Predict test data
        self.predictions = clf.predict(self.X_test)

        # Confusion matrix
        print(confusion_matrix(self.y_test, self.predictions))

        # Score (Accuracy)
        score_metric = Accuracy()
        score = score_metric(self.y_test, self.predictions)
        print(score)

        return self.predictions

    def segmented_fit_predict(self, test_segmentations, train_segmentations, mode='_'):
        self.X_train, self.y_train, _, _ = self.segmented_get_activations_features('train', train_segmentations + 1)
        self.X_test, self.y_test, self.offsets_test, self.noise_test = self.segmented_get_activations_features('test',
                                                                                                               test_segmentations + 1)
        print(self.X_train.shape, self.y_train.shape)
        with parallel_backend('threading', n_jobs=4):
            # Fit to data
            clf = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=10, solver='liblinear'))
            print("Fitting model... " + str(time.strftime("%H:%M:%S", time.localtime())))

            with open(self.model_string + mode + self.region + 'times.txt', 'a') as f:
                f.write("Fitting model... " + str(time.strftime("%H:%M:%S", time.localtime())) + '\n')

            clf.fit(self.X_train, self.y_train)

            with open(self.model_string + mode + self.region + 'times.txt', 'a') as f:
                f.write("Fitting done! " + str(time.strftime("%H:%M:%S", time.localtime())) + '\n')

            # Predict test data
            self.predictions = clf.predict(self.X_test)

            # Confusion matrix
            print(confusion_matrix(self.y_test, self.predictions))

            # Score (Accuracy)
            score_metric = Accuracy()
            score = score_metric(self.y_test, self.predictions)
            print(score)

        return self.predictions

    def psychometric_threshold(self, values, threshold=0.75):
        pts = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        lsq = least_squares(self.residues, x0=1, args=(pts, values))
        print("Converged parameter:", lsq.x)
        threshold_value = self.t_inverse_cdf(1 - threshold, lsq.x)
        print("Threshold value at " + str(threshold) + ":", threshold_value)
        return threshold_value, lsq.x

    def psychometric_fit(self, mode='verniers', plots=False):
        offsets_dict = {}
        correct_dict = {}
        noise_levels = np.array(['low', 'medium', 'high'])
        errors = np.zeros(7)
        thresholds = np.zeros(7)

        for i in range(len(noise_levels)):
            for sample in range(self.y_test.size):
                if self.noise_test[sample] == noise_levels[i]:
                    if self.y_test[sample] == 0:  # left
                        test_string = str(self.y_test[sample]) + "_" + str(self.offsets_test[sample])
                        if test_string in offsets_dict:
                            offsets_dict[test_string] += 1
                        else:
                            offsets_dict[test_string] = 1
                        if self.predictions[sample] == 0:  # correct
                            if test_string in correct_dict:
                                correct_dict[test_string] += 1
                            else:
                                correct_dict[test_string] = 1
                    if self.y_test[sample] == 1:  # right
                        test_string = str(self.y_test[sample]) + "_" + str(self.offsets_test[sample])
                        if test_string in offsets_dict:
                            offsets_dict[test_string] += 1
                        else:
                            offsets_dict[test_string] = 1
                        if self.predictions[sample] == 1:  # correct
                            if test_string in correct_dict:
                                correct_dict[test_string] += 1
                            else:
                                correct_dict[test_string] = 1

            left_percentages = np.zeros(self.offsets_test.max() + 1)
            right_percentages = np.zeros(self.offsets_test.max() + 1)

            for j in range(self.offsets_test.max() + 1):
                left_percentages[j] = correct_dict[str(0) + "_" + str(j)] / offsets_dict[str(0) + "_" + str(j)]
                right_percentages[j] = correct_dict[str(1) + "_" + str(j)] / offsets_dict[str(1) + "_" + str(j)]

            psychometric_values = np.concatenate((1 - np.flip(left_percentages), right_percentages))

            threshold_value, parameter = self.psychometric_threshold(psychometric_values)

            points = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            rmse = np.sum((psychometric_values - self.t_cdf(points, scale=parameter)) ** 2 / 18) ** (1 / 2)
            print("Root mean square error:", rmse)
            errors[i] = rmse
            thresholds[i] = threshold_value
            print(errors, thresholds)
            if plots:
                plt.plot(points, psychometric_values)
                plt.plot(points, self.t_cdf(points, scale=parameter))
                plt.savefig(self.model_string + mode + self.region + '_' + str(noise_levels[i]) + '.png')
                plt.clf()
            offsets_dict = {}
            correct_dict = {}
        psychometric_results = pd.DataFrame([thresholds, errors])
        psychometric_results.to_csv(self.model_string + mode + self.region + str(noise_levels[i]) + '_psychometrics.csv')
        return noise_levels[errors == errors.min()]

    @staticmethod
    def t_cdf(x, scale):
        return t.cdf(x, df=16, scale=scale)

    @staticmethod
    def t_inverse_cdf(p, scale):
        return t.isf(p, df=16, scale=scale)

    @staticmethod
    def residues(scale, x, points):
        return t.cdf(x, df=16, scale=scale) - points


# Method 1: Create new dataset
# unc = Uncrowding('alexnet', 'IT', [(0, 170)])
# unc.make_verniers_stimulus_set(400, 'mini_verniers', mode='verniers', split=0.5)
# unc.fit_predict()
# print(unc.psychometric_fit())

# Method 2: Use pre-existing dataset
# unc = Uncrowding('resnet-18', 'IT', [(0, 170)],
#                  train_metadata='mini_verniers_train.csv', test_metadata='mini_verniers_test.csv',
#                  directory='mini_verniers')
# unc.fit_predict()
# print(unc.psychometric_fit())

# for brain_region in ['V1', 'V2', 'V4', 'IT']:
#     for mode in ['verniers','crowding','uncrowding']:
#         unc = Uncrowding('alexnet', brain_region, [(0, 170)])
#         unc.make_verniers_stimulus_set(400, brain_region + '_' + mode, mode=mode, split=0.5)
#         unc.fit_predict()
#         print(unc.psychometric_fit())

region = 'V4'
unc = Uncrowding('alexnet', region, [(0, 170)])
unc.set_directory('crowding')
test_seg = 20
train_seg = 80
unc.segmented_fit_predict(test_seg, train_seg, mode='uncrowding')
print(unc.psychometric_fit(mode='uncrowding', plots=True))

