import os
import shutil
import cv2
import random
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure

height = 480
width = 640

dirs_f1 = '/home/wilfred/Datasets/Motion/final_3/f1'
dirs_f2 = '/home/wilfred/Datasets/Motion/final_3/f2'
dirs_f3 = '/home/wilfred/Datasets/Motion/final_3/f3'
dirs_f4 = '/home/wilfred/Datasets/Motion/final_3/f4'
dest = '/home/wilfred/Datasets/Motion/final_processed_512_512'
num = 4

def curate_f1():
    
    dirs = sorted(glob.glob(dirs_f1+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))
        
        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]
            ##addition
            dest_f = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]+'_'+'f'
            if not os.path.exists(dest_):
                os.mkdir(dest_)
                ##addition
                os.mkdir(dest_f)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2
                    h_c = im.shape[0]//2
                    im = im[:,w_c-h_c:w_c+h_c]
                    im = cv2.resize(im, (512,512), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
                    ##addition
                    im_f = cv2.flip(im, 1)
                    cv2.imwrite(dest_f+'/'+img.split('/')[-1], im_f)
    return None

def curate_f2():

    dirs = sorted(glob.glob(dirs_f2+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]
            ##addition
            dest_f = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]+'_'+'f'
            if not os.path.exists(dest_):
                os.mkdir(dest_)
                ##addition
                os.mkdir(dest_f)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2
                    h_c = im.shape[0]//2
                    im = im[:,w_c-h_c:w_c+h_c]
                    im = cv2.resize(im, (512,512), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
                    ##addition
                    im_f = cv2.flip(im, 1)
                    cv2.imwrite(dest_f+'/'+img.split('/')[-1], im_f)
    return None

def curate_f3():

    dirs = sorted(glob.glob(dirs_f3+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]
            ##addition
            dest_f = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]+'_'+'f'
            if not os.path.exists(dest_):
                os.mkdir(dest_)
                ##addition
                os.mkdir(dest_f)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2 
                    h_c = im.shape[0]//2
                    im = im[:,w_c-h_c:w_c+h_c]
                    im = cv2.resize(im, (512,512), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
                    ##addition
                    im_f = cv2.flip(im, 1)
                    cv2.imwrite(dest_f+'/'+img.split('/')[-1], im_f)
    return None

def curate_f4():

    dirs = sorted(glob.glob(dirs_f4+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]
            ##addition
            dest_f = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]+'_'+'f'
            if not os.path.exists(dest_):
                os.mkdir(dest_)
                ##addition
                os.mkdir(dest_f)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2
                    h_c = im.shape[0]//2

                    im = im[:,w_c-h_c:w_c+h_c]
                    im = cv2.resize(im, (512,512), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
                    ##addition
                    im_f = cv2.flip(im, 1)
                    cv2.imwrite(dest_f+'/'+img.split('/')[-1], im_f)
    return None

def compileImagefiles(source,dest):
    dirs = sorted(glob.glob(source+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/*'))
        for i in range(len(files)//6):
            images = files[i*6:i*6+6]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    shutil.copy(img, dest_)
    return None

def convert(path):
    list_categories = sorted(glob.glob(path+'/*'))
    
    for l in list_categories:
        
        movies = glob.glob(l+'/*')
        
        for lf in movies:
            if not os.path.exists(lf[:-4]):
                os.makedirs(lf[:-4])
                extract(lf)
    return None

def resizeImage(path):

    files = sorted(glob.glob(path+'/*'))
    
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img[:,:288,:], (96,96))
        fname = os.path.splitext(f)[0][-10:]
        print(fname)
        cv2.imwrite(path+'/'+fname+'_scaled.jpg', img)
    return None

def extract(path):

    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0

    while success:
        cv2.imwrite(path[:-4]+'/'+'frame%s.jpg'%str(count).zfill(5), image)
        success, image = video.read()
        print('Frame read : ',success)
        count += 1
    return None

def process():    
    dirs_process = sorted(glob.glob(dirs+'/*'))

    for sub in dirs_process:
        lst = glob.glob(sub+'/*')
        print(len(lst))
    return None

def datasetCreator():
    dir_='/home/wilfred/Datasets/Motion/VIRAT'
    lst = glob.glob(dir_+'/*')
    print(lst)
    for l in lst:
        st = str(os.path.splitext(l)[0].split('/')[-1])
        video = cv2.VideoCapture(l)
        success, image = video.read()
        count = 0    
        print(dir_+'/'+st)
        os.mkdir(dir_+'/'+st)
        while success:
            
            cv2.imwrite(dir_+'/'+st+'/frame%s.jpg'%str(count).zfill(5), image)
            success, image = video.read()
            count += 1
    return None

def test_create():

    t_dirs = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data'

    lst = sorted([f for f in glob.glob(t_dirs+'/*') if os.path.isdir(f)])
    
    for i in range(len(lst)):
        l = sorted(glob.glob(lst[i]+'/*'))
        for j in l:
            count = 0
            im = cv2.imread(j,0)
            if im.shape[0] % 2 == 1:
                w_c = im.shape[1]
                h_c = im.shape[0] - 1
                im = im[:h_c,w_c-h_c:w_c]
            else:
                w_c = im.shape[1]
                h_c = im.shape[0]
                im = im[:,w_c-h_c:w_c]
            im = cv2.resize(im, (120,120), interpolation = cv2.INTER_AREA)
            temp = j.split('/')[-1].split('.')[0]
            cv2.imwrite(lst[i]+'/'+temp+'-'+str(count)+'.jpg',im)
            count += 1
        
        '''
            im = cv2.imread(input_imgs[j],0)
            if im.shape[0] % 2 == 1:
                w_c = im.shape[1]
                h_c = im.shape[0] - 1
                im = im[:h_c,w_c-h_c:w_c]
            else:
                w_c = im.shape[1]
                h_c = im.shape[0]
                im = im[:,w_c-h_c:w_c]
            im = cv2.resize(im, (120,120), interpolation = cv2.INTER_AREA)
            imgs.append(im.reshape(width, height, 1))

        input_images[0] = np.concatenate(imgs,axis=2)
        input_images[0] /= 255.

        output_image = model_new.predict(input_images)
        arr = output_image[0]
        new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        cv2.imwrite(d+'/predicted.jpg',new_arr)
        '''
if __name__ == "__main__":

    curate_f1()
    curate_f2()
    curate_f3()
    curate_f4()
