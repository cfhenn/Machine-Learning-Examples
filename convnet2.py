import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

#set to true to train data, false to load weights and run
#them on test data
TRAIN_DATA = False

n_classes = 2
batch_size = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

#All these variables are used to hold weights and biases
#so they can be saved for later use
W1 = tf.Variable(tf.random_normal([3,3,3,1,32]), name='W1')
W2 = tf.Variable(tf.random_normal([3,3,3,32,64]), name='W2')
Wfc = tf.Variable(tf.random_normal([54080,1024]), name='Wfc')
Wout = tf.Variable(tf.random_normal([1024, n_classes]), name='Wout')

B1 = tf.Variable(tf.random_normal([32]), name='B1')
B2 = tf.Variable(tf.random_normal([64]), name='B2')
Bfc = tf.Variable(tf.random_normal([1024]), name='Bfc')
Bout = tf.Variable(tf.random_normal([n_classes]), name='Bout')

#This saver is how tensorflow saves variables
saver = tf.train.Saver({'W1s': W1, 'W2s':W2, 'Wfcs':Wfc, 'Wouts':Wout,
                        'B1s': B1, 'B2s':B2, 'Bfcs':Bfc, 'Bouts':Bout})

#how tensorflow does 3d convolution
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

#how tensorflow decides on the window to convolute
def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

#This is what implements the cnn. 2 convolution layers
#and one fully connected layer
def convolutional_neural_network(x,save):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    output = None

    #If we want to save the data, we don't want to run the
    #code again
    if save == False:
        #prepare input for convolution
        x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

        #convolution layer 1
        conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool3d(conv1)


        #convolution layer 2
        conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool3d(conv2)

        #fully connected layer
        fc = tf.reshape(conv2,[-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)

        #final output layer
        output = tf.matmul(fc, weights['out'])+biases['out']

    #save means we want the values to be stored. This
    #happens at the end of a training session.
    if save == True:
        W1.assign(weights['W_conv1'])
        W2.assign(weights['W_conv2'])
        Wfc.assign(weights['W_fc'])
        Wout.assign(weights['out'])

        B1.assign(biases['b_conv1'])
        B2.assign(biases['b_conv2'])
        Bfc.assign(biases['b_fc'])
        Bout.assign(biases['out'])

    return output

#responsible for training the cnn
def train_neural_network(x):

    #this provides the prediction
    prediction = convolutional_neural_network(x,False)

    #This identifies the cost associated with the guess
    #How bad is the guess off from the actual answer
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=y) )

    #This tries to reduce cost as much as possible
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=cost)
    
    #How many epochs
    hm_epochs = 1 

    #This starts the tensor flow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
       
        #For each epoch...
        for epoch in range(hm_epochs):
            epoch_loss = 0

            #For each data row in train_data
            for data in train_data:
                total_runs += 1
                try:
                    #seperate data from label
                    X = data[0]
                    Y = data[1]

                    #run the optimization code
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    print('epoch_loss',epoch_loss,'epoch:',epoch)
                    successful_runs += 1
                except Exception as e:
                    pass
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            #this identifies if the guess is correct
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            #this reports accuracy
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')

        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
       
        #The True means "store the weights and biases" in
        #variables
        convolutional_neural_network(x,True)

        #This saves the weights and bias values in a seperate
        #file
        save_path = saver.save(sess, "tf.model")
        print("Model saved in file: %s" % save_path)

#If TRAIN_DATA is true, this code trains the data, else it will run
#a save CNN on other data
if TRAIN_DATA:
    #Get the trainingData
    train_dir = './trainingData/'
    patients = os.listdir(train_dir)
    much_data = []
    for patient in patients:
        filename = train_dir + patient
        much_data.append(np.load(filename))

    #seperate out training and validation data
    train_data = much_data[:-2]
    validation_data = much_data[-2:]

    #run the cnn training function
    train_neural_network(x)

else:
    #get the testingData
    test_dir = './testingData/'
    patients = os.listdir(test_dir)
    much_data = []
    for patient in patients:
        filename = test_dir + patient
        temp_dat = np.load(filename)
        much_data.append([temp_dat,patient])

    total_cnt = 0
    correct_cnt = 0
    
    #make the file that'll hold guesses
    guess_file = open('guesses.txt','w')

    x = tf.Variable(tf.random_normal([20,50,50]))
    x2 = tf.Variable(tf.random_normal([1,50,50,20,1]))
   
    #start session
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        #this brings back the weights and biases from
        #the stored model file.
        saver.restore(sess, "tf.model")

        #for each data row in much_data
        for data in much_data:
            #This all recreates the original cnn function

            #near the top of the file
            #----------------------------------------------------------------------------------
            sess.run(tf.assign(x,data[0]))

            x2 = sess.run(tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1]))

            conv1 = sess.run(tf.nn.relu(conv3d(x2, W1) + B1))
            conv1 = sess.run(maxpool3d(conv1))

            conv2 = sess.run(tf.nn.relu(conv3d(conv1, W2) + B2))
            conv2 = sess.run(maxpool3d(conv2))

            fc = sess.run(tf.reshape(conv2,[-1, 54080]))
            fc = sess.run(tf.nn.relu(tf.matmul(fc, Wfc)+Bfc))
            fc = sess.run(tf.nn.dropout(fc, keep_rate))

            output = sess.run(tf.matmul(fc, Wout)+Bout)
            #--------------------------------------------------------------------------------------

            #use output to identify accuracy and output the
            #guess next to the patient id on a txt file
            guess = np.argmax(output)
            #correct = np.argmax(data[0][1])

            guess_file.write('%s,%s\n' % (str(data[1][:-4]), str(guess)))
            
            #if guess == correct:
            #    correct_cnt += 1
            #total_cnt += 1

            print(output, 'guess', guess, 'correct', correct)
    print('accuracy',correct_cnt/float(total_cnt))
    guess_file.close()

