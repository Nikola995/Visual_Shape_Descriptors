#%%
#Generate classes used
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial.distance as dist

from OS_S_Classes import *

import os, os.path
import time

#%%
#import the pictures
def import_images(databaseDirectory):
    image_list = []
    for listed_image in os.listdir(databaseDirectory):
        #get the path for every image in database
        imagePath = os.path.join(databaseDirectory,listed_image)
        #using VideoCapture because images are in gif format
        r = cv2.VideoCapture(imagePath)
        retflag,retimg = r.read()
        #stop the program if there wasn't an image loaded
        assert retflag is True
        #merge data for image in one class
        #the second argument will be the image's metadata
        #while removing the imageDirectory from the string
        image = Image(retimg,imagePath[len(databaseDirectory)+1:])
        #add image to list
        image_list.append(image)
        #release the VideoCapture for the next gif image
        r.release()
    return image_list

#using a separate directory for the query and the rest of the images
database = import_images("MPEG7dataset")

#extract query images from database by taking 1 image from every class
def split_database(database,img_num):
    query = []
    new_database = []
    for image in database:
        if image.image_class_num is img_num:
            query.append(image)
        else:
            new_database.append(image)
    return new_database,query

#generate query
database,query = split_database(database,1)
#%%
#print all the images
# =============================================================================
# 
# for image in database:
#     plt.axis("off")
#     plt.imshow(image.image,cmap="gray")
#     plt.show()
# 
# =============================================================================
#%%
#Define all the descriptor methods
# =============================================================================
# #function used for additional preprocessing before computing moments
# def process_segmented_image(image,show=False):
#         image = clean_border(image,3)
#         seg_floodfill = image.copy()
#         # Mask used to flood filling.
#         # Notice the size needs to be 2 pixels than the image.
#         h, w = image.shape[:2]
#         mask = np.zeros((h+2, w+2), np.uint8)
#         
#         # Floodfill from point (0, 0)
#         cv2.floodFill(seg_floodfill, mask, (0,0), 0);
#         # Combine the two images to get the foreground.
#         im_out = cv2.bitwise_xor(image,seg_floodfill)
#         #show thresholded and processed image side by side
#         if show:
#             show_side_by_side(image, im_out)
#         return im_out
# =============================================================================
import mahotas
def find_minEnclosingCircle(image):
    ret2,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #find the contours in the image
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #find the biggest contour (the one with the most contour points)
    cnt = sorted(contours, key = np.size, reverse = True)[0]
    #get the center coordinates and radius of the minEnclosingCircle
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    #convert them to integer values
    center = (int(x),int(y))
    radius = int(radius)
    return center,radius

def ZernikeDescriptor(image):
    #convert the image color format to 
    #one compatible with the descriptor function
    cvtimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #find the center of the biggest object and
    #the radius of the minimal Enclosing Circle
    center,radius = find_minEnclosingCircle(cvtimage)
    #calculate the zernike moments using the center
    #and radius of the minimal Enclosing Circle            
    start_time = time.process_time()
    descriptor = mahotas.features.zernike_moments(cvtimage,
                                                  radius,
                                                  cm=center)
    end_time = time.process_time()
    compute_time= end_time - start_time
    
    return descriptor, compute_time


from shape_context import ShapeContext
SCDescriptor = ShapeContext()

def ShapeContextDescriptor(image):
    #find the contours of the object
    #in the image
    image_contours = SCDescriptor.get_points_from_img(image)
    #Compute the descriptor
    start_time = time.process_time()
    descriptor = SCDescriptor.compute(image_contours)
    end_time = time.process_time()
    compute_time= end_time - start_time
    
    return sum(descriptor), compute_time

from css import CurvatureScaleSpace
CSSDescriptor = CurvatureScaleSpace()

def CurvatureScaleSpaceDescriptor(image):
    #find the contours of the object
    #in the image
    image_contours = CSSDescriptor.get_points_from_img(image)
    #Compute the descriptor
    start_time = time.process_time()
    css = CSSDescriptor.generate_css(image_contours,
                                       image_contours.shape[1],
                                       0.01)
    vcs = CSSDescriptor.generate_visual_css(css,9)
    end_time = time.process_time()
    compute_time= end_time - start_time
    
    return vcs, compute_time

from pyefd import elliptic_fourier_descriptors
#%%
import math
def FourierDescriptor(image):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #ret2,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #find the contours of the object
    #in the image
    cvtimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #find the contour points of the image
    image_contours = SCDescriptor.get_points_from_img(cvtimage)
    
    #find the centroid of the contour
    centroid,_ = find_minEnclosingCircle(cvtimage)
    #Compute the descriptor
    start_time = time.process_time()
    #find the distances between the centroid and all the contour points
    #to make the descriptor translation invariant
    dist_c = []
    for con_point in image_contours:
        radius = dist.euclidean(centroid,tuple(con_point))
        theta = math.atan2(con_point[1]-centroid[1],con_point[0]-centroid[0])
        dist_c.append((radius,theta))
    dist_c = np.asarray(dist_c)
    dist_c = np.transpose(dist_c)
    
    
    descriptor = elliptic_fourier_descriptors(dist_c,
                                              order=10)
    
    end_time = time.process_time()
    compute_time= end_time - start_time
    
    #The output of the descriptor function does not
    #create a feature vector, so we have to modify the output
    #return descriptor.flatten()[3:], compute_time
    return descriptor.flatten(), compute_time
    #return descriptor[:int(np.ceil(len(descriptor)/2))], compute_time

#%%
#compute descriptors
#change later to include whole query and database
for query_image in query:
    query_image.add_descriptor(ZernikeDescriptor)
    query_image.add_descriptor(ShapeContextDescriptor)
    query_image.add_descriptor(CurvatureScaleSpaceDescriptor)
    query_image.add_descriptor(FourierDescriptor)

for database_image in database:
    database_image.add_descriptor(ZernikeDescriptor)
    database_image.add_descriptor(ShapeContextDescriptor)
    database_image.add_descriptor(CurvatureScaleSpaceDescriptor)
    database_image.add_descriptor(FourierDescriptor)

#%%
FourierSimilarityQuery = Similarity_Query(query,database,"FourierDescriptor")
apFD,arFD = FourierSimilarityQuery.precision_recall_graph()

ZernikeSimilarityQuery = Similarity_Query(query,database,"ZernikeDescriptor")
apZD,arZD = ZernikeSimilarityQuery.precision_recall_graph()

SCSimilarityQuery = Similarity_Query(query,database,"ShapeContextDescriptor")
apSC,arSC = SCSimilarityQuery.precision_recall_graph()

CSSDSimilarityQuery = Similarity_Query(query,database,"CurvatureScaleSpaceDescriptor")
apCSSD,arCSSD = CSSDSimilarityQuery.precision_recall_graph()
#%%
#Graph all the precision-recall curves together
def graph(precision,recall,name):
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall,precision,label=name)
    plt.legend()

graph(apFD,arFD,"FD")
graph(apZD,arZD,"ZD")
graph(apSC,arSC,"SCD")
graph(apCSSD,arCSSD,"CSSD")

#%%
#Get average times
FDdatatime = get_average_descriptor_time(database,"FourierDescriptor")
FDquerytime = get_average_descriptor_time(query,"FourierDescriptor")
print("Average time for Fourier Descriptor")
print("Database"+str(FDdatatime))
print("Query"+str(FDquerytime))
ZDdatatime = get_average_descriptor_time(database,"ZernikeDescriptor")
ZDquerytime = get_average_descriptor_time(query,"ZernikeDescriptor")
print("Average time for Zernike Moment Descriptor")
print("Database"+str(ZDdatatime))
print("Query"+str(ZDquerytime))
SCDdatatime = get_average_descriptor_time(database,"ShapeContextDescriptor")
SCDquerytime = get_average_descriptor_time(query,"ShapeContextDescriptor")
print("Average time for Shape Context Descriptor")
print("Database"+str(SCDdatatime))
print("Query"+str(SCDquerytime))
CSSDdatatime = get_average_descriptor_time(database,"CurvatureScaleSpaceDescriptor")
CSSDquerytime = get_average_descriptor_time(query,"CurvatureScaleSpaceDescriptor")
print("Average time for Curvature Scale Space Descriptor")
print("Database"+str(CSSDdatatime))
print("Query"+str(CSSDquerytime))