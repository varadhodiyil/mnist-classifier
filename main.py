from __future__ import print_function
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from  sklearn.model_selection import  KFold
import  numpy as np
import datetime
from sklearn.model_selection import train_test_split
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def csv_images(file_name):
    df = pd.read_csv(file_name)
    y_train = df.label
    # df = df.drop(['label'],axis=1)
    x = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # (42000,28,28,1) array
    y = df.iloc[:, 0].values  # (42000,1) array
    # plt.figure(figsize=(15, 9))
    # plt.title(y_train_valid_labels[1])
    # plt.imshow(x_train_valid[1].reshape(28,28), cmap=cm.binary)
    # plt.waitforbuttonpress()
    labels_count = np.unique(y).shape[0]
    y = dense_to_one_hot(y, labels_count).astype(np.uint8)
    return x, y

class NET(object):
    def __init__(self,learing_rate=0.001,batch_size=50,epoch=10):
        self.learning_rate = learing_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.current_epoch = 0
        self.index_in_epoch =0
        self.perm_array = np.array([])
        self.n_log_step = 0
        self.keep_prob = 0.33
        self.log_step = 0.2
        self.learn_rate_pos = 0
        self.model_name = "v1"

    def add_summary(self, var, var_name):
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def set_X_Train(self,x_train):
        self.X_Train = x_train

    def set_Y_Train(self,y_train):
        self.Y_Train = y_train

    def next_batch(self, iter):

        '''
        Return a total of `num` random samples and labels. 
        '''
        # num = self.batch_size
        # data = self.x_train
        # labels = self.y_train
        # idx = np.arange(0 , len(data))
        # np.random.shuffle(idx)
        # idx = idx[:num]
        # data_shuffle = data[idx]
        # labels_shuffle = labels[idx]
        start = iter * self.batch_size
        end = iter* self.batch_size + self.batch_size
        total_len = self.x_train.shape[0]
        
        data_x = self.x_train[start:end]
        data_y = self.y_train[start:end]
        # labels_shuffle = np.asarray(labels_shuffle.values.reshape(len(labels_shuffle), 1))

        return data_x, data_y

    def next_mini_batch(self):

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        self.current_epoch += self.batch_size / len(self.x_train)

        # adapt length of permutation array
        if not len(self.perm_array) == len(self.x_train):
            self.perm_array = np.arange(len(self.x_train))

        # shuffle once at the start of epoch
        if start == 0:
            np.random.shuffle(self.perm_array)

        # at the end of the epoch
        if self.index_in_epoch > self.x_train.shape[0]:
            np.random.shuffle(self.perm_array)  # shuffle data
            start = 0  # start next epoch
            self.index_in_epoch = self.batch_size  # set index to mini batch size

        end = self.index_in_epoch
        x_tr = self.x_train[self.perm_array[start:end]]
        y_tr = self.y_train[self.perm_array[start:end]]

        return x_tr, y_tr



    def model(self):
        #params
        filter_size = 5
        output_size = 36

        conv_strides = [1, 1, 1, 1]
        max_pool_strides = [1,2,2,1]
        padding = 'SAME'
        fc_size = output_size* 4*4

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')
        input_size = self.X.get_shape().as_list()[-1]

        # Layer 1 + Weights + bias + max_pool
        weights = tf.Variable(tf.random_normal([filter_size, filter_size, 1, output_size]),
                              name='conv_1_weights')
        self.add_summary(weights, "conv_1_weights")
        tf.summary.histogram(weights.name, weights)
        biases = tf.Variable(tf.random_normal([output_size]), name='conv_biases_1')
        self.add_summary(biases, "conv_biases_1")
        conv = tf.nn.conv2d(self.X, weights, strides=conv_strides, padding=padding) + biases
        conv_1 = tf.nn.relu(conv,name="convolution_1")
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=max_pool_strides,padding='SAME', name = "Pool_1")



        #Layer 2 Weights + bias + max pool

        weights_2 = tf.Variable(tf.random_normal([filter_size, filter_size, output_size, output_size]),
                                    name='conv_biases_2')
        tf.summary.histogram(weights_2.name, weights_2)
        self.add_summary(weights_2,"conv_biases_2")
        biases_2 = tf.Variable(tf.random_normal([output_size]), name='conv_biases2')
        self.add_summary(biases_2,"conv_biases2")

        conv = tf.nn.conv2d(pool_1, weights_2, strides=conv_strides, padding=padding) + biases
        conv_2 = tf.nn.relu(conv, name="convolution_2")
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=max_pool_strides, padding='SAME', name="Pool_2")



        #Layer 3 Weights + bias + max_pool

        weights = tf.Variable(tf.random_normal([filter_size, filter_size, output_size, output_size]),
                              name='conv_weights_3')
        self.add_summary(weights,"conv_weights_3")

        tf.summary.histogram(weights.name, weights)
        biases = tf.Variable(tf.random_normal([output_size]), name='conv_biases_3')
        self.add_summary(biases,"conv_biases_3")

        conv = tf.nn.conv2d(pool_2, weights, strides=conv_strides, padding=padding) + biases
        conv_3 = tf.nn.relu(conv, name="convolution_3")
        pool_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=max_pool_strides, padding='SAME', name="Pool_3")


        # Fully Connected Layer

        weights = tf.Variable(tf.random_normal([output_size*4*4, fc_size]),
                              name='fc_weights')
        self.add_summary(weights,'fc_weights')

        tf.summary.histogram(weights.name, weights)
        biases = tf.Variable(tf.random_normal([fc_size]), name='fc_biases')
        self.add_summary(biases,"fc_biases")

        pool_3_flat = tf.reshape(pool_3, [-1, 4 * 4 * output_size],name='fc_flat')  # (.,1024)
        fc_layer = tf.nn.relu(tf.matmul(pool_3_flat, weights) + biases, name='fc_layer')


        #Drop Out:
        self.keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob')
        fc_dropout = tf.nn.dropout(fc_layer, self.keep_prob_tf,name = 'h_fc1_drop_tf')

        weights = tf.Variable(tf.random_normal([fc_size, 10]), name='weights_fc_2')
        self.add_summary(weights,"weights_fc_2")

        biases = tf.Variable(tf.random_normal([10]), name='fc_2_biases')
        self.add_summary(biases,"fc_2_biases")

        z_pred_tf = tf.add(tf.matmul(fc_dropout, weights),biases, name='z_pred_tf')  # => (.,10)

        # cost function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.Y, logits=z_pred_tf), name='cross_entropy')
        self.learn_rate_tf = tf.placeholder(dtype=tf.float32, name="learn_rate_tf")
        self.optimizer_step = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(self.cross_entropy, name='train_step')

        self.Y_pred_proba = tf.nn.softmax(z_pred_tf, name='Y_PRED_PROB')
        pred_correct = tf.equal(tf.argmax(self.Y_pred_proba, 1),tf.argmax(self.Y, 1),
                                     name='y_pred_correct_tf')

        self.accuracy = tf.reduce_mean(tf.cast(pred_correct, dtype=tf.float32),
                                          name='accuracy')


        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('accuracy_tf', self.accuracy)
        self.summary_meged = tf.summary.merge_all()

    def train(self,X_train, Y_train):
        tf.reset_default_graph()
        self.model()
        cv_num = 10  # cross validations default = 20 => 5% validation set
        kfold = KFold(cv_num, shuffle=True, random_state=123)
        learn_rate_step_size = 3

        learn_rate_array = [10 * 1e-4, 7.5 * 1e-4, 5 * 1e-4, 2.5 * 1e-4, 1 * 1e-4, 1 * 1e-4,
                                 1 * 1e-4, 0.75 * 1e-4, 0.5 * 1e-4, 0.25 * 1e-4, 0.1 * 1e-4,
                                 0.1 * 1e-4, 0.075 * 1e-4, 0.050 * 1e-4, 0.025 * 1e-4, 0.01 * 1e-4,
                                 0.0075 * 1e-4, 0.0050 * 1e-4, 0.0025 * 1e-4, 0.001 * 1e-4]
        train_loss_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                         name='train_loss_tf', validate_shape=False)
        valid_loss_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                         name='valid_loss_tf', validate_shape=False)
        train_acc_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                        name='train_acc_tf', validate_shape=False)
        valid_acc_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                        name='valid_acc_tf', validate_shape=False)
        filepath = os.path.join(os.getcwd(), self.model_name)
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        with tf.Session() as sess:
            start = datetime.datetime.now()
            saver = tf.train.Saver()
            # for i, (train_index, valid_index) in enumerate(kfold.split(X_train)):
            if True:
                self.x_train , x_valid, self.y_train , y_valid = train_test_split(X_train,Y_train,test_size=0.3,random_state=40)
                # train and validation data of original images
                # self.x_train = X_train[train_index]
                # self.y_train = Y_train[train_index]
                # x_valid = X_train[valid_index]
                # y_valid = Y_train[valid_index]

                current_epoch = 0
                batch_size_per_epoch = self.x_train.shape[0] / self.batch_size
                print('learnrate = ', self.learning_rate, ', n_epoch = ', 1,
                      ', mb_size = ', batch_size_per_epoch)
                tf.global_variables_initializer().run()
                
                print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'), ': start training')
                print("Total steps : %d " % (int(self.epoch * batch_size_per_epoch) + 1) )
                self.train_writer = tf.summary.FileWriter(os.path.join(filepath, 'train'), sess.graph)
                self.valid_writer = tf.summary.FileWriter(os.path.join(filepath, 'valid'), sess.graph)
                # for i in range(int(self.epoch * batch_size_per_epoch) + 1):
                for current_epoch in range(1,self.epoch+1):
                    learn_rate_pos = int(self.current_epoch // learn_rate_step_size)

                    if not self.learning_rate == learn_rate_array[learn_rate_pos]:
                        self.learning_rate = learn_rate_array[learn_rate_pos]
                        print(datetime.datetime.now() - start, ': set learn rate to %.6f' % self.learning_rate)

                    for j in range(batch_size_per_epoch):
                        x_batch, y_batch = self.next_batch(j)
                        sess.run(self.optimizer_step, feed_dict={self.X: x_batch,
                                                                self.Y: y_batch,
                                                                self.keep_prob_tf: self.keep_prob,
                                                                self.learn_rate_tf: self.learning_rate})

                        # if k % int(self.log_step * batch_size_per_epoch) == 0 or k == int(self.epoch * batch_size_per_epoch):

                        self.n_log_step += 1  # for logging the results

                        feed_dict_train = {
                                            self.X: x_batch,
                                            self.Y: y_batch,
                                            self.keep_prob_tf: 1.0,
                                        }

                        feed_dict_valid = {
                                            self.X: x_valid,
                                            self.Y: y_valid,
                                            self.keep_prob_tf: 1.0
                                        }

                        train_summary = sess.run(self.summary_meged, feed_dict=feed_dict_train)
                        self.train_writer.add_summary(train_summary, j*current_epoch)
                        train_loss.append(sess.run(self.cross_entropy,
                                                feed_dict=feed_dict_train))

                        train_acc.append(self.accuracy.eval(session=sess,
                                                            feed_dict=feed_dict_train))
                        print('epoch %d : step : %d train loss = %.4f, train acc = %.4f ' % (
                                current_epoch, j, train_loss[-1],train_acc[-1] ) )
                        # For every tenth Iter run Validation 
                        if j%10 == 0:
                            valid_summary = sess.run(self.summary_meged, feed_dict=feed_dict_valid)
                        
                            self.valid_writer.add_summary(valid_summary, j*current_epoch)

                        

                            valid_loss.append(sess.run(self.cross_entropy,
                                                    feed_dict=feed_dict_valid))

                            valid_acc.append(self.accuracy.eval(session=sess,
                                                                feed_dict=feed_dict_valid))

                            print('epoch %d : step : %d train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f' % (
                                current_epoch, j, train_loss[-1], valid_loss[-1],
                                train_acc[-1], valid_acc[-1]))
                        if j % 100 == 0:
                            print("Saving model")
                            saver.save(sess,filepath+"/model/"+self.model_name,j*current_epoch)
                tl_c = np.concatenate([train_loss_tf.eval(session=sess), train_loss], axis=0)
                vl_c = np.concatenate([valid_loss_tf.eval(session=sess), valid_loss], axis=0)
                ta_c = np.concatenate([train_acc_tf.eval(session=sess), train_acc], axis=0)
                va_c = np.concatenate([valid_acc_tf.eval(session=sess), valid_acc], axis=0)

                sess.run(tf.assign(train_loss_tf, tl_c, validate_shape=False))
                sess.run(tf.assign(valid_loss_tf, vl_c, validate_shape=False))
                sess.run(tf.assign(train_acc_tf, ta_c, validate_shape=False))
                sess.run(tf.assign(valid_acc_tf, va_c, validate_shape=False))

                print('running time for training: ', datetime.datetime.now() - start)
                

                saver.save(sess,filepath+"/model/"+self.model_name)



x, y = csv_images("train.csv")
n = NET()
n.train(x,y)


