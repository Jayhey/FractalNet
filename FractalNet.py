import argparse
import time
import os
from sklearn.preprocessing import OneHotEncoder
from fractal_module import *



epochs = 30
batch_size = 64
b = 5
c = 4

logs_path = os.path.join(os.getcwd(), "output")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_true = y_test

one_hot_encoder = OneHotEncoder(sparse=False)
y_train, y_test = one_hot_encoder.fit_transform(y_train), one_hot_encoder.fit_transform(y_test)

filters = [64, 128, 256, 512, 512]
dropout_train = [0., 0.1, 0.2, 0.3, 0.4][:b]
dropout_test = [0., 0., 0., 0., 0.][:b]

train_img_tensor, train_label_tensor = set_input(x_train, y_train)
test_img_tensor, test_label_tensor = set_input(x_test, y_test)

train_imgs = tf.data.Dataset.from_tensor_slices((train_img_tensor, train_label_tensor))
test_imgs = tf.data.Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))  # , test_drop_tensor))
infer_imgs = tf.data.Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))

train_imgs = train_imgs.batch(batch_size).shuffle(buffer_size=1000).repeat()
test_imgs = test_imgs.batch(batch_size).shuffle(buffer_size=1000).repeat()
infer_imgs = infer_imgs.batch(batch_size).repeat(1)

train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()

handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()

drop_holder = tf.placeholder(tf.bool, shape=[b, c, 2 ** (c - 1)])
learning_rate = tf.placeholder(tf.float32)
dropout_holder = tf.placeholder(tf.float32, shape=[b, ])
print("Tensor preprocessing complete")

out = x
block_list = []
for i in range(b):
    block_list.append(FractalBlock(x=out,
                                    c=c,
                                    block_num=i,
                                    filters=filters[i],
                                    drop_holder=drop_holder,
                                    drop_rate=dropout_holder[i]))
    out = block_list[i].tensor_col[0][-1]
    out = tf.keras.layers.MaxPool2D()(out)
flat = tf.layers.Flatten()(out)

logits = tf.layers.Dense(y.shape[1])(flat)

global_step = tf.Variable(0, trainable=False, name='global_step')
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

momentum = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step=global_step)

# lr_decay = tf.train.exponential_decay(learning_rate, global_step, 500, 0.9, staircase=True)
# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(extra_update_ops):
# 	adam = tf.train.AdamOptimizer(lr_decay).minimize(loss, global_step=global_step)
# 	sgd = tf.train.GradientDescentOptimizer(lr_decay).minimize(loss, global_step=global_step)
# 	rms = tf.train.RMSPropOptimizer(lr_decay).minimize(loss, global_step=global_step)
# 	momentum = tf.train.MomentumOptimizer(lr_decay, momentum=0.9).minimize(loss, global_step=global_step)

y_prob = tf.nn.softmax(logits)
y_pred = tf.argmax(y_prob, 1)

correct_prediction = tf.equal(y_pred, tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("accuray", accuracy)
tf.summary.scalar("loss", loss)

merged_summary_op = tf.summary.merge_all()

model_name = "Fractal_Network_Cifar10"

start_time = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
test_handle = sess.run(test_iterator.string_handle())
infer_handle = sess.run(infer_iterator.string_handle())

train_writer = tf.summary.FileWriter(os.path.join(logs_path, 'train'))#, sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(logs_path, 'test'))

print("Session ready!")

train_batches = len(x_train) // batch_size + 1
test_batches = len(x_test) // batch_size + 1

LEARNING_RATE = 0.02
remain_epochs = epochs
test_drop = np.ones((b, c, 2 ** (c - 1))).astype(bool)
for i, r in enumerate(reversed(range(epochs))):
    print("\r-------{} Epoch--------".format(i + 1))
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    sess.run(infer_iterator.initializer)
    train_drop = set_drop(b, c, x_train, batch_size)
    for j in range(train_batches):
        summary, _, acc, loss_ = sess.run([merged_summary_op, momentum, accuracy, loss],
                                            feed_dict={handle: train_handle,
                                                        drop_holder: train_drop[j],
                                                        learning_rate: LEARNING_RATE,
                                                        dropout_holder: dropout_train})

        step = tf.train.global_step(sess, global_step)
        sys.stdout.write("\rTraining Iter : {}, Acc : {:.3f}, Loss : {:.4f}".format(step, acc, loss_))
        sys.stdout.flush()

        if j % 100 == 0:
            train_writer.add_summary(summary, step)
            summary, acc, loss_ = sess.run([merged_summary_op, accuracy, loss],
                                            feed_dict={handle: test_handle,
                                                        drop_holder: test_drop,
                                                        dropout_holder: dropout_test})
            print("\nValidation Iter : {}, Acc : {:.3f}, Loss : {:.4f}".format(step, acc, loss_))
            test_writer.add_summary(summary, step)

    total_pred = np.empty((0,))
    for j in range(test_batches):
        try:
            pred, acc, loss_ = sess.run([y_pred, accuracy, loss],
                                        feed_dict={handle: infer_handle,
                                                    drop_holder: test_drop,
                                                    dropout_holder: dropout_test})
            pred = pred.flatten()
            total_pred = np.append(total_pred, pred).flatten()
        except:
            break
    true = y_true.flatten().astype(int)
    infer_acc = sum(np.equal(total_pred, true)) / len(total_pred) * 100
    print("\nTotal test accuracy : {:.4f}%".format(infer_acc))

    if r == remain_epochs // 2:
        remain_epochs = int(remain_epochs / 2)
        LEARNING_RATE = LEARNING_RATE / 10

    print("")
print("-----------End of training-------------")

end_time = time.time() - start_time

print("{} seconds".format(end_time))

# sess.run(test_iterator.initializer)
# total_loss = []
# for j in range(test_batches):
#     acc, loss_ = sess.run([accuracy, loss],
#                           feed_dict={handle: test_handle,
#                                      drop_holder: test_drop})
#     total_loss.append(loss_)
#     sys.stdout.write("\nValidation Iter : {}, Acc : {:.3f}, Loss : {:.4f}".format(step, acc, loss_))
#     sys.stdout.flush()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Training FractalNet. Set epochs, batch size, block and column")
#     parser.add_argument('-E', '--epoch', type=int,
#                         help="Number of epochs(default : 200)", default=200)
#     parser.add_argument('-B', '--batch', type=int,
#                         help="batch size(default : 100)", default=100)
#     parser.add_argument('-b', '--block', type=int,
#                         help="Number of fractal blocks. (default : 5, max : 5)", default=5)
#     parser.add_argument('-c', '--column', type=int,
#                         help="Number of columns in one block. (default : 4)", default=4)

#     args = parser.parse_args()

#     epochs = args.epoch
#     batch_size = args.batch
#     b = args.block
#     c = args.column

#     main(epochs, batch_size, b, c)
