'''
SeamCarving.py - content-aware image resizing algorithm based on Seam Carving
Copyright (C) 2012 Minh Phan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy
import scipy.misc as scm
from scipy import ndimage
import time
import sys

def compute_energy(im):
    im = rgb2gray(im)
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0)
    dy = ndimage.sobel(im, 1)
    mag = numpy.hypot(dx,dy)
    mag *= 255.0/numpy.max(mag)
    return mag

def rgb2gray(img):
    R = 0.2989*img[:,:,0]
    G = 0.5870*img[:,:,1]
    B = 0.1140*img[:,:,2]
    
    tmp = numpy.add(R,G)
    return numpy.add(tmp,B)


def compute_min_energy(img):
    height,width = img.shape
    out = numpy.empty((height,width))
    out.fill(0)
    out[0][:] = img[0][:]
    
    for i in xrange(1,height):
        for j in xrange(0,width):
            if j-1 < 0:
                out[i,j] = img[i,j] + min(out[i-1,j], out[i-1,j+1])
            elif j+1 >= width:
                out[i,j] = img[i,j] + min(out[i-1,j-1], out[i-1,j])
            else:
                out[i,j] = img[i,j] + min(out[i-1,j-1], out[i-1,j], out[i-1,j+1])
    return out
    
def find_optimal_seam(img):
    height,width = img.shape
    s = numpy.zeros((height,1))
    for i in xrange(height-1,-1,-1):
        if i == height-1:
            value = min(img[i,:])
            j = numpy.where(img[i][:]==value)[0][0]
        else:
            if s[i+1,0] == 0:
                tmp = [float("Inf"), img[i,s[i+1,0]], img[i,s[i+1,0]+1]]
            elif s[i+1,0] == width -1:
                tmp = [img[i,s[i+1,0]-1], img[i,s[i+1,0]], float("Inf")]
            else:
                tmp = [img[i,s[i+1,0]-1], img[i,s[i+1,0]], img[i,s[i+1,0]+1]]
            value = min(tmp)
            tmp_j = tmp.index(value)
            tmp_j -= 1
            j = s[i+1,0] + tmp_j
        s[i,0] = j
    return s

def remove_seam(img,seam):
   height,width,dim = img.shape
   out = numpy.zeros((height,width-1,dim))
   for i in xrange(0,dim):
       for j in xrange(0,height):
           if seam[j,0] == 0:
               out[j,:,i] = img[j,1:width,i]
           elif seam[j,0] == width-1:
               out[j,:,i] = img[j,0:width-1,i]
           else:
               y = numpy.append(img[j,0:seam[j,0],i],img[j,seam[j,0]+1:width,i])
               out[j,:,i] = y
   return out
    #rows, cols = img.shape[:2]
    #return numpy.array([img[i,:][numpy.arange(cols) != seam[i]] for i in range(rows)])

def color_seam(im,seam):
    height,width,dim = im.shape
    for i in xrange(0,len(seam)):
        im[i][seam[i][0]][0] = 255;
        im[i][seam[i][0]][1] = 0;
        im[i][seam[i][0]][2] = 0;
    return im       

def main():
    start = time.time()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    desired_w = sys.argv[3]
    desired_h = sys.argv[4]
    image = scm.imread(infile)
    height,width,dim = image.shape
    num_iterations_w = width - int(desired_w)
    num_iterations_h = height - int(desired_h)
    count = 0
    while (count < num_iterations_w):#image.shape[1] > desired_w):
        print "width: ", count, "seams removed"
        print "image dimensions: ", image.shape[1], image.shape[0]
        sobel = compute_energy(image)
        min_eng = compute_min_energy(sobel)
        seam = find_optimal_seam(min_eng)
        image = remove_seam(image,seam)
        #image = color_seam(image,seam)
        count += 1

    count = 0
    image = image.transpose(1,0,2)
    print image.shape
    while (count < num_iterations_h):
        print "height: ", count, "seams removed"
        print "image dimensions: ", image.shape[1], image.shape[0]
        sobel = compute_energy(image)
        min_eng = compute_min_energy(sobel)
        seam = find_optimal_seam(min_eng)
        image = remove_seam(image,seam)
        count += 1
    scm.imsave(outfile, image.transpose(1,0,2))
    print "Done"
    print time.time() - start

if __name__ == '__main__':
    main()