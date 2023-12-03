from cProfile import label
from lib2to3.pytree import convert
from turtle import delay
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import sys
import pandas as pd
import pickle
import scipy.io
# from torch import batch_norm

from tqdm import tqdm

from wrl_utils import wrl_to_numpy,find_img_bounderies,trim_img,image_resize
import open3d as o3d
import glob,os

# from sklearn.model_selection import train_test_split
import sklearn.model_selection as skl_ms
from sklearn.preprocessing import StandardScaler
from sklearn import discriminant_analysis 

from pathlib import Path

import argparse
# import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

DEBUG = False


def uniqueHash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

class Preproccesor(object):

    def __init__(self,data_set_folder,n_pcl_points,is_mat_files,min_files=None,filter_ratios=None) -> None:
        super().__init__()


        self.is_mat_files = is_mat_files
        self.train_folder = os.path.join(data_set_folder,'train')
        self.test_folder = os.path.join(data_set_folder,'test')

        self.filter_ratios = filter_ratios

        self.train_files,self.train_labels = self.get_files_matrix_and_labels(self.train_folder,min_files)
        self.test_files,self.test_labels = self.get_folder_files_and_labels(self.test_folder)

        # print(len(self.test_labels),self.test_labels)

        if DEBUG : 
            print("train files")
            print(self.train_files,self.train_labels)
            print("test files")
            print(self.test_files,self.test_labels)

        self.min_points = n_pcl_points

        self.prepare_pcls()


        if DEBUG : 
            print("cloud points")
            print(self.full_train_pcl,self.full_test_pcl,self.full_train_pcl.shape,self.full_test_pcl.shape)


    def prepare_pcls(self):
        self.full_train_pcl,self.full_test_pcl = self.create_point_cloud_test_train_matrix(self.train_files,[self.test_files],min_points=self.min_points)
        self.full_test_pcl = self.full_test_pcl.reshape(-1)

    def get_folder_files_and_labels(self,folder_path):
        files_list = glob.glob(os.path.join(folder_path,'*.wrl'))
        if(self.is_mat_files):
            files_list = glob.glob(os.path.join(folder_path,'*.mat'))
        else:
            #---------------------------------------------------------------------USE FILTER BY RATIO------------------------------------------------------
            files_list = [self.filter_by_ratio(f) for f in files_list]
            files_list = [f for f in files_list if not f is None]

        
        files_list.sort()
        labels = [f.split(os.sep)[-1].replace('.wrl','') for f in files_list]
        if(self.is_mat_files):
            [f.split(os.sep)[-1].replace('.mat','') for f in files_list]

        return np.array(files_list),np.array(labels) 
        

    def get_files_matrix_and_labels(self,path,min_files=None):
        labels = []
        files = []
        folders = os.listdir(path)
        folders.sort()

        if min_files is None:
            min_files = 99999
            for folder in folders:
                files_list = glob.glob(os.path.join(path,folder,'*.wrl'))
                if(self.is_mat_files):
                    files_list = glob.glob(os.path.join(path,folder,'*.mat'))
                n_files = len(files_list)
                if n_files < min_files: min_files = n_files

        for folder in folders:
            files_list = glob.glob(os.path.join(path,folder,'*.wrl'))
            if(self.is_mat_files):
                files_list = glob.glob(os.path.join(path,folder,'*.mat'))
            
            if len(files_list) < min_files: continue

            n_files = min_files
            rnd_indecies = np.random.choice(len(files_list),n_files,replace=False)
            files_list = [files_list[i] for i in rnd_indecies]

            labels.append([str(folder)]*n_files)
            files.append(files_list)

        if DEBUG:print(len(files_list),np.array(labels))

        return np.array(files),np.array(labels)

    def create_numpy_cloud_point(self,img,thr = 0.0):

        #-----------------------------------------------------------------------------IMAGE NORMALIZATION---------------------------------------------------------------------
        grad_thr=250
        boundaries = find_img_bounderies(img,grad_thr=grad_thr)
        img = trim_img(img.copy(),boundaries)


        #--------------------------------------------------------------------------POINT CLOUD NORMALIZATION-----------------------------------------------------------------------
        h,w = img.shape

        r,c = np.where(img > np.mean(img))
        # cy = 100
        # cx = 133/2.0
        cy = h/2.0
        cx = w/2.0
        y_cm = 8
        x_cm = 5.5
        cp = np.array([np.array([(i-cy)/h*y_cm,(j-cx)/w*x_cm,img[i,j]/1000.0]) for i,j in zip(r,c)],dtype=object)

        
        return cp

    def filter_by_ratio(self,file,thr=350):
        #-----------------------------------------------------------------------------FILTER BY RATIO---------------------------------------------------------------------
        if self.filter_ratios is None: return file
        img = wrl_to_numpy(file)
        
        grad_thr=thr
        boundaries = find_img_bounderies(img,grad_thr=grad_thr)
        img_trimed = trim_img(img.copy(),boundaries)

        h,w = img_trimed.shape
        # print(h,w)
        ratio = h/w
        low_ratio,high_ratio = self.filter_ratios
        if ratio < low_ratio or ratio > high_ratio:
            # plt.imshow(img_trimed)
            # plt.pause(15)
            # raise Exception("Sample fildered - ratio : {:.1f},  range : {}".format(ratio,self.filter_ratios))
            print("Sample fildered - ratio : {:.1f},  range : {}, file :{}".format(ratio,self.filter_ratios,file))
            return None
        return file




    def create_pcd_from_numpy(self,vec3d_arr,n_min_points = None):
        if not n_min_points is None:
            indecies = np.random.choice(vec3d_arr.shape[0], n_min_points)
            vec3d_arr = vec3d_arr[indecies,:]
        vec3d_arr = vec3d_arr - np.mean(vec3d_arr,axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vec3d_arr)

        return pcd

    def create_point_cloud_test_train_matrix(self,train_files_matrix,test_files_matrix,equal_n_points=True,min_points = None):
        # train_np_data_2d = [np.asarray([self.create_numpy_cloud_point(wrl_to_numpy(f)) for f in cluster_files],dtype=object) for cluster_files in train_files_matrix]
        # test_np_data_2d = [np.asarray([self.create_numpy_cloud_point(wrl_to_numpy(f)) for f in cluster_files],dtype=object) for cluster_files in test_files_matrix]

        train_np_data_2d = []#[np.asarray([self.create_numpy_cloud_point(wrl_to_numpy(f)) for f in cluster_files],dtype=object) for cluster_files in train_files_matrix]
        test_np_data_2d = []#[np.asarray([self.create_numpy_cloud_point(wrl_to_numpy(f)) for f in cluster_files],dtype=object) for cluster_files in test_files_matrix]

        with tqdm(total=len([0 for cf in train_files_matrix for _ in cf]), desc="Train wrl to numpy") as pbar:
            for cluster_files in train_files_matrix:
                tmp = []
                for f in cluster_files:
                    if(self.is_mat_files):
                        tmp.append(scipy.io.loadmat(f)['vertices'])
                    else:
                        tmp.append(self.create_numpy_cloud_point(wrl_to_numpy(f)))
                    pbar.update(1)
                train_np_data_2d.append(np.asarray(tmp,dtype=object))

        with tqdm(total=len([0 for cf in test_files_matrix for _ in cf]), desc="Test wrl to numpy") as pbar:
            for cluster_files in test_files_matrix:
                tmp = []
                exclude_labels = []
                for i,f in enumerate(cluster_files):
                    if(self.is_mat_files):
                        # print(f)
                        try:
                            tmp.append(scipy.io.loadmat(f)['vertices'])
                        except Exception as e:
                            print('ERROR line 217 ({}), file: {}'.format(e,f))
                    else:
                        np_image = wrl_to_numpy(f)
                        #-----------------------------------------------------------------------------FILTER BY RATIO---------------------------------------------------------------------
                        # try:
                        #     if not self.filter_ratios is None: self.filter_by_ratio(np_image)
                        #     tmp.append(self.create_numpy_cloud_point(np_image))
                        # except Exception as e:
                        #     print('\n',i,e,' File :{}'.format(f))
                        tmp.append(self.create_numpy_cloud_point(np_image))
                    pbar.update(1)
                test_np_data_2d.append(np.asarray(tmp,dtype=object))


        
        
        train_np_data_2d = np.asarray(train_np_data_2d,dtype=object)
        test_np_data_2d = np.asarray(test_np_data_2d,dtype=object)

        full_train_pc = np.empty(train_np_data_2d.shape,dtype=object)
        full_test_pc = np.empty(test_np_data_2d.shape,dtype=object)
        
        if min_points is None:
            # print(train_np_data_2d.shape,test_np_data_2d.shape)
            min_train = np.min([np.min([s.shape[0] for s in c]) for c in train_np_data_2d])
            min_test = np.min([np.min([s.shape[0] for s in c]) for c in test_np_data_2d])
            min_points = np.min([min_test,min_train])
            # min_points = np.min([np.min([s.shape[0] for s in c]) for c in np.concatenate((train_np_data_2d,test_np_data_2d))])



        with tqdm(total=full_train_pc.shape[0]*full_train_pc.shape[1],desc="Train") as pbar:
            for class_id in range(full_train_pc.shape[0]):
                for sample_id in range(full_train_pc.shape[1]):
                    sample = train_np_data_2d[class_id,sample_id]
                    if equal_n_points:
                        full_train_pc[class_id,sample_id] = self.create_pcd_from_numpy(sample,min_points)
                    else:
                        full_train_pc[class_id,sample_id] = self.create_pcd_from_numpy(sample)
                    pbar.update(1)

        with tqdm(total=full_test_pc.shape[0]*full_test_pc.shape[1], desc="Test") as pbar:
            # for class_id in tqdm(range(full_test_pc.shape[0]),desc="Test"):
            for class_id in range(full_test_pc.shape[0]):
                for sample_id in range(full_test_pc.shape[1]):
                    sample = test_np_data_2d[class_id,sample_id]
                    if equal_n_points:
                        full_test_pc[class_id,sample_id] = self.create_pcd_from_numpy(sample,min_points)
                    else:
                        full_test_pc[class_id,sample_id] = self.create_pcd_from_numpy(sample)  
                    pbar.update(1)

        self.min_points = min_points
        return (full_train_pc,full_test_pc)

class TournamentTrainer(Preproccesor) : 

    def __init__(self,**kwargs) -> None:
        # print(kwargs) 

        self.data_set_folder = kwargs['data_set_folder']
        self.models_folder = kwargs['models_folder']
        self.results_folder = kwargs['results_folder']
        self.temp_folder = kwargs['temp_folder']
        self.n_iterations = kwargs['n_iterations']
        self.n_pcl_points = kwargs['n_pcl_points']
        self.random_class_selection = kwargs['random_class_selection']
        self.batch_size = kwargs['batch_size']

        self.is_mat_files = kwargs['is_mat_files']
        self.min_files = kwargs['min_files']

        self.filter_ratios = kwargs['filter_ratios']

        self.seeds_queue = list(range(100*100))

        super().__init__(self.data_set_folder,self.n_pcl_points,self.is_mat_files,self.min_files,self.filter_ratios)

        self.n_pcl_points = self.min_points

        print(self.n_pcl_points)

        self.train_pcl_dict = self.convert_mat_to_dict(self.full_train_pcl,self.train_labels)
        
    

        if DEBUG:
            print("train_pcl_dict")
            print(self.train_pcl_dict)
            print("statistics_dict")
            print(self.statistics_dict)

    def run_classifier_with_retrain(self):

        random_class_selection = self.random_class_selection
        batch_size = self.batch_size
        n_iterations = self.n_iterations

        list_of_classes = list(self.train_pcl_dict.keys())
        list_of_classes.sort()

        # layer_split_keys = self.split_to_index_barches(batch_size=batch_size,list_of_classes=list_of_classes,to_return=[],random=random_class_selection)
        # models_dict_layer_1 = self.train_models_set(layer_split_keys,self.train_pcl_dict)

        d_progress = 1.0# / (len(self.full_test_pcl) * n_iterations)
        total_iters = len(self.full_test_pcl) * n_iterations

        self.statistics_dict = self.create_statistics_dict(self.test_labels,self.train_pcl_dict.keys())
        # print(self.statistics_dict)
        
        # print(self.full_test_pcl.shape,n_iterations)

        with tqdm(total=total_iters,desc="Tournament iterations") as pbar:

            for iteration in range(n_iterations):
                
                # if not random_class_selection : np.random.seed(self.seeds_queue.pop(0))
                layer_split_keys = self.split_to_index_barches(batch_size=batch_size,list_of_classes=list_of_classes,to_return=[],random=random_class_selection)
                models_dict_layer_1 = self.train_models_set(layer_split_keys,self.train_pcl_dict)

                for pcl_sample,sample_lbl in zip(self.full_test_pcl,self.test_labels):

                    pred = self.tournament_classify_sample(np.array([pcl_sample]),
                                                            models_dict_layer_1,
                                                            self.train_pcl_dict,
                                                            layer_split_keys,
                                                            random_class_selection=random_class_selection,
                                                            batch_size=batch_size)
                    pbar.update(d_progress)

                    self.statistics_dict[sample_lbl][pred[0]] += 1
                    self.statistics_dict[sample_lbl]['last_iteration'] += 1

                    self.save_stat_dict(self.statistics_dict,self.test_labels,self.train_pcl_dict.keys())

                self.prepare_pcls()
                
                self.train_pcl_dict = self.convert_mat_to_dict(self.full_train_pcl,self.train_labels)
                # layer_split_keys = self.split_to_index_barches(batch_size=batch_size,list_of_classes=list_of_classes,to_return=[],random=random_class_selection)
            
                # models_dict_layer_1 = self.train_models_set(layer_split_keys,self.train_pcl_dict)
        
        # print(self.statistics_dict)

                    # print(pred)
        # print(layer_split_keys)

    def _run_classifier(self):

        random_class_selection = self.random_class_selection
        batch_size = self.batch_size
        n_iterations = self.n_iterations

        list_of_classes = list(self.train_pcl_dict.keys())

        # if not random_class_selection : np.random.seed(self.seeds_queue.pop(0))
        layer_split_keys = self.split_to_index_barches(batch_size=batch_size,list_of_classes=list_of_classes,to_return=[],random=random_class_selection)
        
        models_dict_layer_1 = self.train_models_set(layer_split_keys,self.train_pcl_dict)

        d_progress = 1.0# / (len(self.full_test_pcl) * n_iterations)
        total_iters = len(self.full_test_pcl) * n_iterations

        self.statistics_dict = self.create_statistics_dict(self.test_labels,self.train_pcl_dict.keys())
        # print(self.statistics_dict)
        
        # print(self.full_test_pcl.shape,n_iterations)

        with tqdm(total=total_iters,desc="Tournament iterations") as pbar:

            for iteration in range(n_iterations):

                # if not random_class_selection : np.random.seed(self.seeds_queue.pop(0))

                for pcl_sample,sample_lbl in zip(self.full_test_pcl,self.test_labels):

                    # print(pcl_sample)

                    pred = self.tournament_classify_sample(np.array([pcl_sample]),
                                                            models_dict_layer_1,
                                                            self.train_pcl_dict,
                                                            layer_split_keys,
                                                            random_class_selection=random_class_selection,
                                                            batch_size=batch_size)
                    pbar.update(d_progress)

                    self.statistics_dict[sample_lbl][pred[0]] += 1
                    self.statistics_dict[sample_lbl]['last_iteration'] += 1

                    self.save_stat_dict(self.statistics_dict,self.test_labels,self.train_pcl_dict.keys())
        
        print(self.statistics_dict)

                    # print(pred)
        # print(layer_split_keys)


    def train_models_set(self,list_of_sets,train_dict):
        models_dict = {}
        
        for _,_set in enumerate(list_of_sets):
            set_dict = {key: train_dict[key] for key in _set}
            pc_class_matrix = np.asarray(list(set_dict.values()))
            
            labels = [[_set[i]]*pc_class_matrix[i].shape[0] for i in range(pc_class_matrix.shape[0])]
            labels = np.asarray(labels).reshape(-1)        
            if DEBUG : print("train_models_set: set : {}\nmatrix shape : {}\n# labels : {}\n set_dict : {}\n".format(_set,pc_class_matrix.shape,len(labels),set_dict))
            models_dict[str(_set)] = self.train_model(pc_class_matrix,labels)

        return models_dict

    def test_model(self,train_pc_class_matrix,test_pc_class_matrix,model_scaler_tuple):

        (model,scaler,_) = model_scaler_tuple
        try:
            (model,scaler,pc_class_matrix) = model_scaler_tuple
            if not pc_class_matrix is None:
                train_pc_class_matrix = pc_class_matrix
        except: pass

        score_mat = self.build_rmse_score_matrix(train_pc_class_matrix.reshape(-1),test_pc_class_matrix.reshape(-1))
        score_mat = np.power(score_mat.copy(),2)

        normalized_score_mat = scaler.transform(score_mat)

        prediction = model.predict(normalized_score_mat)
        return prediction

    def tournament_classify_sample(self,pc_test_sample,models_dict_layer_1,train_pc_dict,layer_split_keys,random_class_selection,batch_size):
        list_of_classes = self.evaluate_models_set(pc_test_sample,train_pc_dict,models_dict_layer_1,layer_split_keys)
        while len(list_of_classes) > pc_test_sample.shape[0] :
            # print("Predictions : {}".format(list_of_classes))
            layer_split_keys = self.split_to_index_barches(batch_size=batch_size,list_of_classes=list_of_classes,to_return=[],random=random_class_selection)
            models_dict = self.train_models_set(layer_split_keys,train_pc_dict)
            list_of_classes = self.evaluate_models_set(pc_test_sample,train_pc_dict,models_dict,layer_split_keys)
        return list_of_classes

    def evaluate_models_set(self,pc_test_sample,train_pc_dict,models_dict,list_of_sets):
        lables = []
        for _,_set in enumerate(list_of_sets):

            model_scaler_tuple = models_dict[str(_set)]

            set_dict = {key: train_pc_dict[key] for key in _set}
            train_pc_class_matrix = np.asarray(list(set_dict.values()))
            if DEBUG : print("Evaluating : {}, train shape {}".format(_set,train_pc_class_matrix.shape))
            prediction = self.test_model(train_pc_class_matrix,pc_test_sample,model_scaler_tuple)
            lables += list(prediction)
        return lables

    def model_name_by_labels(self,labels):

        lbls = list(set(labels))
        lbls.sort()

        model_name = 'LDA_MODEL_{}_{}_.mdl'.format(lbls,self.n_pcl_points)
        model_name = model_name.replace("'",'')
        model_name = model_name.replace(',','_')
        model_name = model_name.replace(' ','')

        scaler_model = model_name.replace("LDA","SCALER")
        
        return model_name,scaler_model

    def load_lda_scaler_models_by_name(self,names_tuple):
        #--------------------------------------------------------------------------ALWAYS RETARAIN (1)-----------------------------------------------------------------------
        # raise Exception("ALWAYS RETRAIN") 
        model_name,scaler_name = names_tuple

        lda_model = None
        scaler_model = None
        with open(os.path.join(self.models_folder,model_name),'rb') as model_file , open(os.path.join(self.models_folder,scaler_name),'rb') as scaler_file:
            lda_model = pickle.load(model_file)
            scaler_model = pickle.load(scaler_file)

        return lda_model,scaler_model

    def save_lda_scaler_models_by_name(self,names_tuple,models_tuple):

        model_name,scaler_name = names_tuple
        lda,scaler = models_tuple

        with open(os.path.join(self.models_folder,model_name),'wb') as model_file , open(os.path.join(self.models_folder,scaler_name),'wb') as scaler_file:
            pickle.dump(lda,model_file,pickle.HIGHEST_PROTOCOL)
            pickle.dump(scaler,scaler_file,pickle.HIGHEST_PROTOCOL)

    def find_outliers_indecies(self,pc_class_matrix,labels,min_cluster_size=7):

        outliers_indecies = []

        pc_matrix = []
        lbls = []

        labels = labels.reshape(pc_class_matrix.shape)

        for idx,cluster in enumerate(pc_class_matrix):
            
            lbl = labels[idx][0]
            cur_dir = os.path.abspath('.')
            score_mat_file_name = os.path.join(cur_dir,'temp','outlier_score_mat_{}.npy'.format(lbl))

            try:
                with open(score_mat_file_name,'rb') as f:
                    indecies = pickle.load(f)
            except:
                score_rmse = self.build_rmse_score_matrix(cluster.reshape(-1),cluster.reshape(-1))
                score_rmse = np.power(score_rmse.copy(),2)

                X = score_rmse

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                pca = PCA()
                pca.fit(X)
                X = pca.transform(X)


                clf = LocalOutlierFactor(n_neighbors=min_cluster_size,novelty=True,p=2)
                clf.fit(X)
                LOF_outliers = clf.predict(X)
                indecies = LOF_outliers
                with open(score_mat_file_name,'wb') as f:
                    pickle.dump(indecies,f)

            pc_matrix += list(cluster[np.array(indecies)==1])
            lbls += list(labels[idx][np.array(indecies)==1])
            
            # outliers_indecies.append(np.array(indecies)==1)
    
        # return np.array(outliers_indecies)
        return np.array(pc_matrix),np.array(lbls)



    def train_model(self,pc_class_matrix,labels):

        names_tuple = self.model_name_by_labels(labels)
        pc_outlier_class_matrix = None
        #-------------------------------------------------------------------------- OUTLIERS ---------------------------------------------------------------
        # pc_outlier_class_matrix,labels = self.find_outliers_indecies(pc_class_matrix,labels) 
        # pc_class_matrix=pc_outlier_class_matrix
        lda = None
        scaler = None
        try:
            lda,scaler = self.load_lda_scaler_models_by_name(names_tuple)
            # print('Model files found! : {}'.format(names_tuple))
            return (lda,scaler,pc_outlier_class_matrix)
        except Exception as e: 
            # print(e,'No model files : {}'.format(names_tuple))
            pass
        # print(labels,pc_class_matrix)

        score_mat = self.build_rmse_score_matrix(pc_class_matrix.reshape(-1),pc_class_matrix.reshape(-1))
        score_mat = np.power(score_mat.copy(),2)

        # Normalization
        scaler = StandardScaler()
        normalized_score_mat = scaler.fit_transform(score_mat)

        # if not lda is None:
        #     return (lda,scaler,pc_outlier_class_matrix)

        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=None)
        lda.fit_transform(normalized_score_mat, labels)
        #--------------------------------------------------------------------------ALWAYS RETARAIN (2)-----------------------------------------------------------------------
        self.save_lda_scaler_models_by_name(names_tuple,(lda,scaler))

        return (lda,scaler,pc_outlier_class_matrix)

    def build_rmse_score_matrix(self,pc_np_array_feautes,pc_np_array_samples):

        # pc_np_array = pc_np_array.reshape(-1)
        score_mat     = np.zeros((pc_np_array_samples.shape[0],
                                    pc_np_array_feautes.shape[0]))

        threshold = 100
        trans_init = np.eye(4)

        n_iterations = range(pc_np_array_samples.shape[0]*pc_np_array_feautes.shape[0])
        iteration = 0
        
        with tqdm(total=len(n_iterations), file=sys.stdout ,disable=not DEBUG) as pbar:
            for i in range(pc_np_array_samples.shape[0]):
                for j in range(pc_np_array_feautes.shape[0]):
                    a = pc_np_array_samples[i]
                    b = pc_np_array_feautes[j]

                    # reg_p2p = o3d.pipelines.registration.registration_icp(a, b, threshold,trans_init,
                    #                                     o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    reg_p2p = o3d.pipelines.registration.evaluate_registration(
                                a, b, threshold, trans_init)
                    # reg_p2p = o3d.registration.registration_icp(a, b, threshold,trans_init,
                    #                     o3d.registration.TransformationEstimationPointToPoint())
                    score_mat[i,j] = reg_p2p.inlier_rmse

                    pbar.set_description('processed: %d' % (1 + iteration))
                    pbar.update(1)
                    iteration+=1

        return score_mat
    
    def stat_dict_file_name(self,test_labels,train_labels):
    
        train_labels.sort()
        test_labels.sort()
        # print(test_labels)
        inner_str = '{}'.format(train_labels,test_labels)

        # print(inner_str)
 
        stat_name = 'STAT_DICT_{}_{}_pts_iter({})_MAT({}).sts'.format(uniqueHash(inner_str),self.n_pcl_points,self.n_iterations,self.is_mat_files)
        stat_name = stat_name.replace("'",'')
        stat_name = stat_name.replace(',','_')
        stat_name = stat_name.replace(' ','')

        return stat_name

    def load_stat_file(self,test_labels,train_labels):
        
        stat_filename = self.stat_dict_file_name(list(test_labels),list(train_labels))
        # print(os.path.join(self.results_folder,stat_filename))
        with open(os.path.join(self.results_folder,stat_filename),'rb') as stat_file:
            statistics_dict = pickle.load(stat_file)
        return statistics_dict

    def save_stat_dict(self,stat_dict,test_labels,train_labels):

        stat_filename = self.stat_dict_file_name(list(test_labels),list(train_labels))
        with open(os.path.join(self.results_folder,stat_filename),'wb') as stat_file:
            pickle.dump(stat_dict,stat_file,pickle.HIGHEST_PROTOCOL)

    def create_statistics_dict(self,test_labels,train_labels):
        statistics_dict = {}
        try:
            statistics_dict = self.load_stat_file(test_labels,train_labels)
            return statistics_dict
        except:
            print("Can't load stat file, creating new...")

        for test_lbl in test_labels:
            statistics_dict[test_lbl] = {"last_iteration" : 0}
            for train_lbl in train_labels:
                statistics_dict[test_lbl][train_lbl] = 0

        return statistics_dict

    def convert_mat_to_dict(self,_pcl,_labels):
        _pcl_dict = {}
        # test_pc_dict = {}

        for class_id in range(_pcl.shape[0]):
            _pcl_dict[_labels[class_id,0]] = _pcl[class_id]

        # for class_id in range(test_pc.shape[0]):
        #     test_pc_dict[test_labels[class_id,0]] = test_pc[class_id]

        # return (train_pc_dict,test_pc_dict)
        return _pcl_dict

    def split_to_index_barches(self,batch_size,list_of_classes,to_return,random=True):

        if not random : np.random.seed(3117)
        # if not random : np.random.seed(self.seeds_queue.pop(0))

        if len(list_of_classes)-1 <= batch_size: 
            to_return.append(list_of_classes)
            return to_return

        # (main,left) = skl_ms.train_test_split(list_of_classes,train_size = batch_size,shuffle=random)
        (main,left) = skl_ms.train_test_split(list_of_classes,train_size = batch_size,shuffle=True)

        to_return.append(main)
        _ = self.split_to_index_barches(batch_size,left,to_return,random)

        # print(to_return)
        return to_return

class TournamentTester(object) : 

    def __init__(self) -> None:
        super().__init__()

# class simple_trainer(object) : 

#     def __init__(self) -> None:
#         super().__init__()

# class simple_tester(object) : 

#     def __init__(self) -> None:
#         super().__init__()




'''
LDA+ICP base seeds clasifiier

optional arguments:
  -h, --help  show this help message and exit
  -d D        Data sets folder containing test and train folders (default "-d
              ./data_set")
  -m M        Models folder to save and load models from (default "-m        
              ./models")
  -r R        Result folder to save results and figures (default "-m
              ./results")
  -t T        Classifier type : (tour) Tournament trainer or (simple) Simple 
              trainer
  -ni NI      Number of iterations (default 100)
  -np NP      Number of cloud points (default 2500)
  -nb NB      Number of classes in each model (default 4)
  -debug      Enable debug mode
  -random     Enable random shuffle of models while training



'''

'''
EXMAPLE 

python app.py -np 2500 -nb 4 -random -ni 1

'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA+ICP base seeds clasifiier, Example : python app.py -np 2500 -nb 4 -random -ni 1')
    parser.add_argument('-d', help='Data sets folder containing test and train folders (default "-d ./data_set")')
    parser.add_argument('-m', help='Models folder to save and load models from  (default "-m ./models")')
    parser.add_argument('-r', help='Result folder to save results and figures (default "-m ./results")')
    parser.add_argument('-t', help='Classifier type : (tour) Tournament trainer or (simple) Simple trainer')
    

    parser.add_argument('-ni', help='Number of iterations (default 100)')
    parser.add_argument('-np', help='Number of cloud points (default 2500)')
    parser.add_argument('-nb', help='Number of classes in each model (default 4)')
    parser.add_argument('-debug', help='Enable debug mode',action='store_true')
    parser.add_argument('-random', help='Enable random shuffle of models while training',action='store_true')
    parser.add_argument('-sts', help='Check current state dict and number of iterations',action='store_true')
    parser.add_argument('-mat', help='Files in the train and test directories are .mat files (matlab)',action='store_true')
    parser.add_argument('-minfiles', help='Minimum files in train folder, otherwise skip folder')
    parser.add_argument('--ratios',nargs='+', help='The ratios range used in order to filter Archiological samples (--ratios <low> <high>)',type=float)
    args = parser.parse_args()



    root_folder = os.path.abspath(os.path.dirname(__file__))#os.path.abspath(os.curdir)
    data_set_folder = os.path.join(root_folder,'data_set') if args.d is None else os.path.join(root_folder,args.d)
    models_folder = os.path.join(root_folder,'models') if args.m is None else os.path.join(root_folder,args.m)
    results_folder = os.path.join(root_folder,'results') if args.r is None else os.path.join(root_folder,args.r)
    temp_folder = os.path.join(root_folder,'temp')

    n_iterations = 100 if args.ni is None else int(args.ni)
    n_pcl_points = None if args.np is None else int(args.np)
    # print(args.random)
    random_class_selection =args.random# False if args.random is None else True
    batch_size = 4 if args.nb is None else int(args.nb)
    is_mat_files=args.mat
    min_files = None if args.minfiles is None else int(args.minfiles)
    ratios = None if args.ratios is None else tuple(args.ratios)

    print(ratios)



    args_dict = vars(args)
    # print(type(args_dict))
    # print(min_files)
    for k,v in args_dict.items():
        print(k,":",v)
        
    # print(vars(args))

    Path(models_folder).mkdir(parents=True, exist_ok=True)
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(temp_folder).mkdir(parents=True, exist_ok=True)

    DEBUG = args.debug
    # exit(0)

    # print(DEBUG,args.debug)
    algo = TournamentTrainer(data_set_folder=data_set_folder
                            ,models_folder=models_folder
                            ,results_folder=results_folder
                            ,temp_folder=temp_folder
                            ,n_iterations = n_iterations
                            ,n_pcl_points = n_pcl_points
                            ,random_class_selection=random_class_selection
                            ,batch_size=batch_size,
                            is_mat_files=is_mat_files,
                            min_files=min_files,
                            filter_ratios=ratios)

    if args.sts:
       
        dict = algo.create_statistics_dict(algo.test_labels,algo.train_pcl_dict.keys())
        print(dict)
        it_dict = {key:val['last_iteration'] for key,val in dict.items()}
        # print({{key:val['last_iteration']} for key,val in dict.items()})
        print(it_dict)
        min_it = min(it_dict.values())
        min_key = list(it_dict.keys())[list(it_dict.values()).index(min_it)]
        print("min iterations - {}:{}".format(min_key,min_it))
    else:
        # algo.run_classifier()
        algo.run_classifier_with_retrain()
        pass



    # if args.t == 'tour':
    #     pass
    # elif args.t == 'simple':
    #     pass
    # else:
    #      args.t = None

    # if args.t is None: 
    #     exit("Error : Classifier type argument is None or not correct!")

def plot_dict(dict):

    n_cols = len(dict.keys())