#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import re
xdim_header = 'xDimension'; ydim_header = 'zDimension'
xdim = 0; ydim = 0

def vrml_to_numpy(file_name):
    holder = []
    with open(file_name,'rb') as vrml:
        for lines in vrml:
            a = re.findall("[-0-9]{1,3}.[0-9]{6}",lines.decode('utf-8'))
            if len(a) == 3:
                holder.append(list(map(float,a)))
    return np.array(holder)

def wrl_to_numpy(file_name):
    
    with open(file_name,'r') as file:
        
        for line in file:
            strings = line.split()
            
            if xdim_header in strings and ydim_header in strings:
                
                xdim = int(strings[strings.index(xdim_header)+1])
                ydim = int(strings[strings.index(ydim_header)+1])
                
                break
                
        data = [next(file).replace('\n','').split() for x in range(ydim)]
    
    data = np.array(data,dtype=np.float)
    data = data.T
    return data
    

if __name__ == '__main__':
    
    #file_path = 'fwdwrl/4A10_BOTTOM.wrl'
    
    #data = wrl_to_numpy(file_path)
     
    #print(data)

    fn = 'WRLJerusalem\\417_10.wrl'
    data = vrml_to_numpy(fn)
    print(data,data.shape)
        


# In[4]:


import cv2

def find_img_bounderies(img,grad_thr = 400,width_gap = 2):

    grad_x = np.gradient(img,axis=1)
    grad_y = np.gradient(img,axis=0)

    right_edge_y,right_edge_x = np.where(grad_x < -grad_thr)
    left_edge_y,left_edge_x = np.where(grad_x > grad_thr)

    top_edge_y,top_edge_x = np.where(grad_y > grad_thr)
    bottom_edge_y,bottom_edge_x = np.where(grad_y < -grad_thr)

    # tuple ([y...],[x...])
    top = [(y,x) for y,x in sorted(zip(top_edge_y,top_edge_x))][0]
    right = [(y,x+width_gap) for x,y in sorted(zip(right_edge_x,right_edge_y),reverse=True)][0]
    bottom = [(y,x) for y,x in sorted(zip(bottom_edge_y,bottom_edge_x),reverse=True)][0]
    left = [(y,x-width_gap) if x >= width_gap else (y,0)  for x,y in sorted(zip(left_edge_x,left_edge_y))][0]

    return (top,right,bottom,left)
    # h,w = img.shape
    # return (top,(0,w),bottom,(0,0))


def trim_img(img,boundaries):
    
    top,right,bottom,left = boundaries
    y1,y2,x1,x2 = top[0],bottom[0],left[1],right[1]
    
    result_img = img[y1:y2,x1:x2]
    return result_img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# In[ ]:
