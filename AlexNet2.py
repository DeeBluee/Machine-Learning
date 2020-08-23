import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle

cls = ['cat', 'dog']
IMG_SIZE = 224
PATH