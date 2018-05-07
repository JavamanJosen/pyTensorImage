#! /usr/bin/env python
# -*- coding: utf-8 -*-

# top 1 accuracy 0.99826 top 5 accuracy 0.99989

import os
import sys
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

reload(sys)

sys.setdefaultencoding('utf-8')


tf.app.flags.DEFINE_integer('charset_size', 3772, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_string('checkpoint_Path', './chineseImage/', "don't find checkpoint_path.")
tf.app.flags.DEFINE_string('checkpoint_dir', '', "don't find checkpoint_dir.")
FLAGS = tf.app.flags.FLAGS

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

#根据siteID 分别赋不同的值

def getParameter(siteId):
    if "40001" == siteId:
        FLAGS.image_size = 64
        FLAGS.charset_size = 3772
    else:
        FLAGS.image_size = 64
        FLAGS.charset_size = 3772



def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')  # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/gpu:5'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                          scope='fc2')
        probabilities = tf.nn.softmax(logits)

        # 返回top k 个预测结果及其概率；返回top K accuracy
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'is_training': is_training,
            'accuracy_top_k': accuracy_in_top_k,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def inference(images, checkpoint_dir):
    print('inference')
    image_set = []
    # 对每张图进行尺寸标准化和归一化
    for image in images:
        #temp_image = Image.open(image).convert('L')
        temp_image = image.convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)

    # allow_soft_placement 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        # 自动获取最后一次保存的模型
        FLAGS.checkpoint_dir = checkpoint_dir
        print checkpoint_dir
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        val_list = []
        idx_list = []
        # 预测每一张图
        for item in image_set:
            temp_image = item
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                                    feed_dict={graph['images']: temp_image,
                                                            graph['keep_prob']: 1.0,
                                                            graph['is_training']: False})
            val_list.append(predict_val)
            idx_list.append(predict_index)
    return val_list, idx_list


# 获取汉字label映射表,跟句checkpointId找到对应的识别中文对应表
def get_label_dict(checkpointId):
    f = open('./chinese_labels', 'r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


def main(images, siteId):
    #获取checkpoint的路径
    checkpoint_dir = FLAGS.checkpoint_Path+"checkpoint/checkpoint"+siteId+"/"
    label_dir = FLAGS.checkpoint_Path+"chineseLabels"+siteId
    label_dict = get_label_dict(label_dir)
    #把要识别的图片标准归一化
    final_predict_val, final_predict_index = inference(images, checkpoint_dir)
    final_reco_text = []  # 存储最后识别出来的文字串
    # 给出top 3预测，candidate1是概率最高的预测
    for i in range(len(final_predict_val)):
        candidate1 = final_predict_index[i][0][0]
        #candidate2 = final_predict_index[i][0][1]
        #candidate3 = final_predict_index[i][0][2]
        final_reco_text.append(label_dict[int(candidate1)])
    print ('=====================OCR RESULT=======================\n')
    # 打印出所有识别出来的结果（取top 1）
    #for i in range(len(final_reco_text)):
     #   print final_reco_text[i],


    return final_reco_text


def domain(images, siteId):
    text = main(images, siteId)
    result = ""
    for i in range(len(text)):
        result = result + text[i]
    return result


if __name__ == "__main__":

    siteId = "40001";
    filename = "/Users/shangzhen/Desktop/jbxx/10017.png";
    img = Image.open(filename)

    getParameter(siteId)

    #传图片,返回的是list集合
    imgList = []
    imgList.append(img)
    result = domain(imgList, siteId)
    print result