# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial.distance as dist

class Image:
    def __init__(self,image,image_path):
        self.image = image
        im_path = image_path.split(".")
        #second part of the image path is image format
        self.image_format = im_path[1]
        #first part of the image path is image name
        im_name = im_path[0].split("-")
        #first part of the image name is image class
        self.image_class = im_name[0]
        #second part of the image name is
        #which image it is in the class
        self.image_class_num = int(im_name[1])
        #descriptors are added by method
        self.descriptors = {}
        
    def add_descriptor(self,descriptor_method):
        descriptor = Image_Descriptor(descriptor_method,self.image)
        self.descriptors[descriptor_method.__name__] = descriptor
    
    def similarity_query_from(self,database,descriptor_method_name,sort=True):
        #Check if image contains descriptor by this method
        assert descriptor_method_name in self.descriptors.keys()
        #Get the image descriptor in a temporary variable
        this_image_desc = self.descriptors[descriptor_method_name]
        
        similarity_list = []
        for image in database:
            #Check if image contains descriptor by this method
            assert descriptor_method_name in image.descriptors.keys()
            dat_image_desc = image.descriptors[descriptor_method_name]
            similarity = compute_similarity(this_image_desc.descriptor,
                                            dat_image_desc.descriptor)
            #Add the similarity to the list with 
            #class comparison between the images
            similarity_list.append((similarity,
                                   self.image_class == image.image_class))
        if sort:
            #Return the sorted list by shortest distance first
            return sorted(similarity_list,key=lambda x: x[0], reverse=False)
        else:
            return similarity_list
    
class Image_Descriptor:
    def __init__(self,descriptor_method,image):
        self.descriptor_method_name = descriptor_method.__name__
        self.descriptor,self.compute_time = descriptor_method(image)

class Similarity_Query:
    def __init__(self,query,database,descriptor_name):
        similarityQueries = []
        for query_image in query:
            query_image_sim_list = query_image.similarity_query_from(database,
                                                                     descriptor_name)
            similarityQueries.append(query_image_sim_list)
        self.sim_lists = similarityQueries
        self.descriptor_method_name = descriptor_name
        self.average_precision = None
        self.average_recall = None
        
    def precision_recall_graph(self,show=False):
        if self.average_precision is not None:
            #plot the precision-recall graph
            plt.plot(self.average_recall,self.average_precision)
            #return the precision-recall values
            return self.average_precision,self.average_recall
        class_lists = []
        #for each similarity query
        for sim_list in self.sim_lists:
            #split the similarity value from the class matching
            val_list,class_list = map(list,zip(*sim_list))
            class_list = np.array(class_list)
            #convert class match from bool to int
            class_lists.append(class_list.astype(int))
        #convert list to np.array
        class_lists = np.array(class_lists)
        
        #generate precision and recall lists for each query in list
        precision_lists = np.zeros((1,class_lists.shape[1]))
        recall_lists = np.zeros((1,class_lists.shape[1]))
        #for each class matching list
        for cl_list in class_lists:
            #append precision list extracted from class list
            precision_lists = np.append(precision_lists,
                                        [precision(cl_list)],
                                        axis=0)
            #append recall list extracted from class list
            recall_lists = np.append(recall_lists,
                                     [recall(cl_list)],
                                     axis=0)
        #get the average precision and recall from all query objects
        average_precision = np.average(precision_lists,axis=0)
        average_recall = np.average(recall_lists,axis=0)
        if show:
            #plot the precision-recall graph
            plt.plot(average_recall,average_precision)
        #save the precision-recall values
        self.average_precision = average_precision
        self.average_recall = average_recall
        #return the precision-recall values
        return average_precision,average_recall
    
def precision(class_list):
    retrieved_relevant = np.cumsum(class_list)
    total = np.arange(1,len(class_list)+1)
    precision_list = retrieved_relevant/total
    
    return precision_list

def recall(class_list):
    retrieved_relevant = np.cumsum(class_list)
    total_relevant = sum(class_list)
    recall_list = retrieved_relevant/total_relevant
    
    return recall_list

#define similarity metric    
def compute_similarity(descriptor1,descriptor2):
    descriptor1 = descriptor1.flatten()
    descriptor2 = descriptor2.flatten()
    
    return dist.canberra(descriptor1,descriptor2)

def get_average_descriptor_time(database,descriptor_name):
    sum_time = 0
    for image in database:
        im_time = image.descriptors[descriptor_name].compute_time
        sum_time += im_time
    return sum_time/len(database)

