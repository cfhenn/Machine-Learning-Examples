import dicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

IMG_PX_SIZE = 50
HM_SLICES = 20.0

data_dir = './SampleImages/'
process_dir = './PreProcess/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('./stage1_labels.csv',index_col=0)


def acquire_slices(patient):
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    return slices
    
    
#print first 20 slices of input image
def print_20_slices(slices):
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:20]):
        y = fig.add_subplot(5,5,num+1)
        y.imshow(each_slice,cmap='gray')
    plt.show()



#resize each slice in slices to a sizeXsize array
def resize(slices,size):
    new_img = []
    for each_slice in slices:

        #does actual resizing
        new_img.append(cv2.resize(np.array(each_slice.pixel_array),(size,size)))

    #nice to keep things as numpy arrays
    new_img = np.array(new_img)
    return new_img

def chunks(l, n):
    #Yield successive n-sized chunks from l.
    for i in range(0, len(l), n):
        yield l[i:i+n]

def mean(l):
    return sum(l) / len(l)

#make number of slices uniformly the same size
def shrink(slices,size):
    new_slices = []
    chunk_sizes = int(math.ceil(len(slices) / float(size)))

    #split slices into chunks of size chunk_size
    for slice_chunk in chunks(slices,chunk_sizes):

        #combine the chunk into one slice
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        slice_chunk = np.array(slice_chunk)

        #put new slice into new_slices
        new_slices.append(slice_chunk)

    #These two if statements fix the fact that new_slices
    #can be a few elements short
    if len(new_slices) is int(size-3):
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    elif len(new_slices) is int(size-2):
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    elif len(new_slices) is int(size-1):
        new_slices.append(new_slices[-1])
    
    #np arrays are nice
    new_slices = np.array(new_slices)

    return new_slices

#put label with data in one array
def package_w_label(slices,label):
    if label == 1:
        new_label = np.array([0,1])
    elif label == 0:
        new_label = np.array([1,0])
    return np.array([slices,new_label])

def preProcess(patient):
    label = labels_df.get_value(patient, 'cancer')
    slices = acquire_slices(patient)
    slices_resize = resize(slices,IMG_PX_SIZE)
    slices_shrunk = shrink(slices_resize,HM_SLICES) 
    final_data = package_w_label(slices_shrunk,label)
    return final_data
    
def save_data(data,patient,directory):
    filename = directory + str(patient)+'.npy'
    np.save(filename,data)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#actual preprocessing happens here
progress = 0
finish = len(patients)

make_dir(process_dir)
failureFile = open('failure.txt','w')

for patient in patients:
    try:
        data = preProcess(patient)
        save_data(data,patient,process_dir) 
        progress += 1
        print str(progress) + ' out of ' + str(finish)
        #print_20_slices(data[0])
        
    except KeyError as e:
        print 'could not preProcess'
        finish -= 1
        failureFile.write("%s\n" % patient)
        

failureFile.close()
         
