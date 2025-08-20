import pickle
import sys
import time 
import numpy as np

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

def load_dataset(dataset="1"):
  cfile = "D:/ece276a/ECE276A_PR1/data/cam/cam" + dataset + ".p"
  ifile = "D:/ece276a/ECE276A_PR1/data/imu/imuRaw" + dataset + ".p"
  vfile = "D:/ece276a/ECE276A_PR1/data/vicon/viconRot" + dataset + ".p"

  ts = tic()
  imud = read_data(ifile)

  if(dataset=="1" or dataset=="2" or dataset=="8" or dataset=="9" or dataset=="10" or dataset=="11"):
    camd = read_data(cfile)
  else:
    camd=0

  if(dataset=="10" or dataset=="11"  ):
    vicd=np.array([0])
  else:
    vicd = read_data(vfile)
  
  toc(ts,"Data import")

  return camd, imud, vicd





