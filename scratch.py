
import csv
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# global variables and hyperparameters
ArchiveList = []
numBatch = 1
epoch = 1


# count describes which of the listed datasets shall be accesed
def read_test_data(count):
    testname = ArchiveList[count]
    testname = str(testname[0])
    filename = '/work/dyn/ctm9918/UCR_ProofOfConcept/UCR_TS_Archive_2015/' + testname + '/' + testname + '_TEST'
#    filename = '/home/daniel/UCR_ProofOfConcept/UCR_TS_Archive_2015/' + testname + '/' + testname + '_TEST'

    x = np.loadtxt(filename, delimiter=',')
    return x


# read the number of classes for the data
def read_num_classes(count):
    return int(ArchiveList[count][1])


def read_training_data(count):
    testname = ArchiveList[count]
    testname = str(testname[0])
    filename = '/work/dyn/ctm9918/UCR_ProofOfConcept/UCR_TS_Archive_2015/' + testname + '/' + testname + '_TRAIN'
#    filename = '/home/daniel/UCR_ProofOfConcept/UCR_TS_Archive_2015/' + testname + '/' + testname + '_TRAIN'
    x = np.loadtxt(filename, delimiter=',')
    return x


# extract labeling data and output a vector with corresponding classes
def get_label_data(data):
    label = data[:, 0]
    vector = np.zeros((len(label), numclasses))   # initialise zero matrix of shape (numExamples, classes)
    for k in range(0, len(label)):
        vector[k][int(label[k])-1] = 1     # at position label[k] in series k set label = 1

    return vector


# extract and shape training data into (numExamples, numTimeSteps, numberOfInputsEachSample)
def get_data(data):
    x = data[0:data.shape[0], 1:data.shape[1]]    # omit the first column
    return x.reshape(x.shape[0], x.shape[1], 1)  # UCR TS Archive consists of only univariate signals


# define model
def define_model(training_data):
    model = Sequential()
    lstm_size_l1 = 500                # size of the hidden layer 1
    model.add(LSTM(lstm_size_l1, input_shape=(training_data.shape[1], 1)))
    # input shape: numTimeSteps, numFeatures per TimeStep

    # output layer for classification
    model.add(Dense(numclasses, activation='sigmoid'))
    return model


# main
# load ArchiveList
with open('/work/dyn/ctm9918/UCR_ProofOfConcept/UCR_ArchiveName.csv', newline='') as csvfile:
# with open('/home/daniel/UCR_ProofOfConcept/UCR_ArchiveName.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        ArchiveList.append(row)


with open('/work/dyn/ctm9918/UCR_ProofOfConcept/Ergebnis.csv', 'w', newline='') as csvfile:
# with open('/home/daniel/UCR_ProofOfConcept/Ergebnis.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    for count in range(0, 84):
        testname = ArchiveList[count]
        testname = str(testname[0])
        print('Training with ' + testname)
        trainingDataSet = read_training_data(count)
        numclasses = read_num_classes(count)
        x = get_data(trainingDataSet)
        y = get_label_data(trainingDataSet)
        model = define_model(x)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # for classification problems
        model.fit(x, y, epochs=epoch, batch_size=numBatch)

        testDataSet = read_test_data(count)
        x_test = get_data(testDataSet)
        y_test = get_label_data(testDataSet)

        score = model.evaluate(x_test, y_test, batch_size=numBatch)

        writer.writerow([testname + ',' + np.array2string(score[0])])


