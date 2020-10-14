#!/usr/bin/python

from __future__ import division
from itertools import count
import os
import cv2
import rospy
import numpy as np
import tensorflow as tf
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError
import argparse
from pathlib import Path
import subprocess
from PIL import Image as PILImage
import errno
from datetime import datetime
from datetime import datetime, timedelta
SPEED = 2.5

DISTANCE = 2
LEFT_ANGLE = 30
RIGHT_ANGLE = 30

TURNING_SPEED = 15
PI = 3.1415926535897
classes = ['Border', 'Box', 'Space', 'Sphere']
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200
def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 

class Counter:
    count = 0


def export_img(image, ds_dir):
    now = datetime.now()
    timestamp = totimestamp(now)
    save_path = "%s/%s%d.png" % (ds_dir, timestamp, Counter.count)
    Counter.count += 1
    image = PILImage.fromarray(image[0], 'RGB')
    image.save(save_path)
    print("Image successfully created to %s" % save_path)

def image_cb(data, ds_dir):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(cv_image)
    if ds_dir != "empty":
        export_img(image, ds_dir)
    else: 
        global graph
        with graph.as_default():
            prediction = classes[np.squeeze(np.argmax(model.predict(image), axis=1))]
            move(prediction)

def preprocess_image(image):
    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return np.resize(resized_image, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

def move(prediction):
    print("[*] " + str(prediction) + " Detected.")
    velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    if prediction == classes[2]:
        print("[*] Moving Straight...")
        move_straight(velocity_publisher)

    elif prediction == classes[1]:
        print("[*] Turning Left...")
        turn_left(velocity_publisher)

    elif prediction == classes[3]:
        print("[*] Turning Right...")
        turn_right(velocity_publisher)

    elif prediction == classes[0]:
        print("[*] Moving Right Back...")
        move_right_back(velocity_publisher)

def move_straight(velocity_publisher):
    vel_msg = Twist()
    vel_msg.linear.x = abs(0.5)
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    current_distance = 0
    t0 = rospy.Time.now().to_sec()

    while (current_distance < DISTANCE):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_distance = SPEED * (t1 - t0)
    vel_msg.linear.x = 0
    velocity_publisher.publish(vel_msg)

def turn_right(velocity_publisher):
    angular_speed = TURNING_SPEED * 2 * PI / 360
    relative_angle = RIGHT_ANGLE* 2 * PI / 360

    vel_msg = Twist()
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    vel_msg.angular.z = -abs(angular_speed)

    current_angle = 0
    t0 = rospy.Time.now().to_sec()

    while (current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)

        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed * (t1 - t0)

    vel_msg.linear.z = 0
    velocity_publisher.publish(vel_msg)

def turn_left(velocity_publisher):
    angular_speed = TURNING_SPEED * 2 * PI / 360
    relative_angle = RIGHT_ANGLE* 2 * PI / 360

    vel_msg = Twist()
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    vel_msg.angular.z = abs(angular_speed)

    current_angle = 0
    t0 = rospy.Time.now().to_sec()

    while (current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)

        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed * (t1 - t0)

    vel_msg.linear.z = 0
    velocity_publisher.publish(vel_msg)

def move_right_back(velocity_publisher):
    vel_msg = Twist()
    vel_msg.linear.x = -2
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 1.8

    velocity_publisher.publish(vel_msg)

def main(export_cls=False, ds_dir=None):
    rospy.init_node('navigator', anonymous=True)
    image_subscriber = rospy.Subscriber("/mybot/camera/image_raw", Image, image_cb, callback_args=ds_dir, queue_size=1, buff_size=2**24)
    try:
        rospy.spin()
    except KeyboardInterrupt as e:
        print("Shutting Down")
        cv2.destroyAllWindows()

def make_directory(parent):
    p = subprocess.Popen("rospack find autonomous_navigation".split(), stdout=subprocess.PIPE, universal_newlines=True)
    ds_dir = p.stdout.readline().replace("\n", "") + "/scripts/datasets/%s" % parent
    try:
        os.makedirs(ds_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return ds_dir

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-for", choices=classes)
    parser.add_argument("--test-for", choices=classes)
    args = parser.parse_args()
    ds_dir = "empty"
    bridge = CvBridge()
    selected_arg = None
    if args.train_for:  
        ds_dir = make_directory("training/%s" % args.train_for)
        export_cls = args.train_for
    elif args.test_for:
        ds_dir = make_directory("test/%s" % args.test_for)
        export_cls = args.train_for
    else:
        model = load_model(os.path.join(os.path.dirname(__file__), "nn_controller.h5"))
        graph = tf.get_default_graph()
    
    main(export_cls=selected_arg, ds_dir=ds_dir)
