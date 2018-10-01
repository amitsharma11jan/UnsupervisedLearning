#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:03:40 2018

@author: Amit Sharma
"""


#python GaussianMixtureMRAmit.py --clusters 2 --dimensions 4 --parameters "/Users/a5sharma/Documents/ISB/DMG/Assignment1/Pratice/MR/IrisInitail.json" "/Users/a5sharma/Documents/ISB/DMG/Assignment1/Pratice/MR/Iris.csv"
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol, JSONValueProtocol, PickleProtocol
from mrjob.step import MRStep
import json
import numpy as np

class Stats():
    def __init__(self,r_sum,r_w_sum,r_w_cov, total):
        self.r_sum = r_sum
        self.r_w_sum = r_w_sum
        self.r_w_cov = r_w_cov
        self.total = total 

def gaussian_pdf(x, mu, cov):
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * np.dot(np.dot((x-mu).T,(np.linalg.inv(cov))),(x-mu))
    return float(part1 * np.exp(part2))
        

def responsibility(x,mu,cov,p,K):
    resps = [p[k]*gaussian_pdf(x,np.array(mu[k]),np.array(cov[k])) for k in range(K)]
    p_x = sum(resps)
    return [float(r_k)/p_x for r_k in resps]
    

def extract_features(line):
    data = line.strip().split(",")
    return [ float(e) for e in data[1:] ]
    
    
def outputFile(key, mixing, means, covar):
    mixing = mixing
    means = means
    covariance = covar
    return {"key":key,"mixing":mixing,"mu":json.dumps(means.tolist()),"covariance":json.dumps(covariance.tolist())}


class GaussianMixtureMR(MRJob):
    
    INPUT_PROTOCOL = RawValueProtocol    
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol
        

    def __init__(self,*args,**kwargs):
        super(GaussianMixtureMR,self).__init__(*args,**kwargs)
        self.resp_sum = [0]*self.clusters
        self.N = 0
        self.resp_w_sum = [np.zeros(self.dim, dtype = np.float64) for i in range(self.clusters)]
        self.resp_w_cov = [np.zeros([self.dim,self.dim], dtype = np.float64) for i in range(self.clusters)]   
        
        
    def configure_args(self):
        super(GaussianMixtureMR,self).configure_args()
        self.add_passthru_arg("--dimensions",
                                    type = int,
                                    help = "dimensionality of input data")
        self.add_passthru_arg("--clusters",
                                    type = int,
                                    help = "number of clusters")
        self.add_passthru_arg("--parameters",
                             type = str,
                             help = "json file with initial parameters")
    
    
    def load_args(self,args):
        super(GaussianMixtureMR,self).load_args(args)
        # number of clusters
        if self.options.clusters is None:
            self.option_parser.error("Please enter number of clusters")
        else:
            self.clusters = self.options.clusters
        # features
        if self.options.dimensions is None:
            self.option_parser.error("Please enter number of column")
        else:
            self.dim = self.options.dimensions
        
        if self.options.parameters is None:
            self.option_parser.error("Initial Parameter are missing..")
            
    def mapper_gmm_init(self):
        with open(self.options.parameters, 'r') as myfile:
            if len(myfile.readlines()) != 0:
                myfile.seek(0)
                params = json.load(myfile)
                self.mu = params["mu"]
                self.covar = params["covariance"]
                self.mixing = params["mixing"]
        
    def mapper_gmm(self,_,line):
        features = extract_features(line)
        assert(len(features)==self.dim), "feature mismatch"
        x = np.array(features)
        r_n = responsibility(x,self.mu,self.covar,self.mixing,self.clusters) 
        self.resp_sum = [self.resp_sum[i]+r_n_k for i,r_n_k in enumerate(r_n)]
        self.resp_w_sum = [w_sum + r_n[i]*x for i,w_sum in enumerate(self.resp_w_sum)]
        self.resp_w_cov = [w_covar+r_n[i]*np.outer(x,x) for i,w_covar in enumerate(self.resp_w_cov)]
        self.N+=1

    def mapper_final_gmm(self):
        matrix_to_list = lambda x: [list(e) for e in x]
        r_sum = self.resp_sum
        r_w_sum = self.resp_w_sum
        r_w_cov = [ matrix_to_list(cov) for cov in self.resp_w_cov]
        for k in range(self.clusters):
            obj = Stats(r_sum[k],r_w_sum[k],r_w_cov[k], self.N)
            yield k, obj

    def reducer_gmm(self,key, values):
        N = 0;
        r_sum = 0
        r_w_sum = np.zeros(self.dim, dtype = np.float64)
        r_w_cov = np.zeros([self.dim,self.dim], dtype = np.float64)
        for value in values:
            r_sum = r_sum + value.r_sum
            r_w_sum = r_w_sum+np.array(value.r_w_sum, dtype = np.float64)
            r_w_cov = r_w_cov + np.array(value.r_w_cov)
            N = N + value.total
        mixing = float(r_sum)/N
        means =  1.0/r_sum*(r_w_sum)
        covar =  1.0/r_sum*(r_w_cov - np.outer(means,means))
        #yield None, '\t'.join([str(key), json.dumps(mixing), json.dumps(means.tolist()), json.dumps(covar.tolist())])
        yield None, outputFile(key, mixing,means,covar)


    def steps(self):
        return [MRStep(mapper_init = self.mapper_gmm_init,
                       mapper = self.mapper_gmm, 
                       mapper_final = self.mapper_final_gmm,
                       reducer = self.reducer_gmm)]
                       
if __name__=="__main__":
    GaussianMixtureMR.run()