## Individual test for original images

==== inception v1 ====           
Defense accuracy: 88.8%          
NT success: 11.2%                
==== inception v2 ====           
Defense accuracy: 91.7%          
NT success: 8.3%                 
==== inception v3 ====           
Defense accuracy: 96.1%          
NT success: 3.9%                 
==== inception v4 ====           
Defense accuracy: 97.2%          
NT success: 2.8%                 
==== inception_resnet_v2 ====    
Defense accuracy: 96.9%          
NT success: 3.1%                 
==== resnet_v1_101 ====          
Defense accuracy: 92.9%          
NT success: 7.1%                 
==== resnet_v1_152 ====          
Defense accuracy: 93.2%          
NT success: 6.8%                 
==== resnet_v2_101 ====          
Defense accuracy: 96.2%          
NT success: 3.8%                 
==== resnet_v2_152 ====          
Defense accuracy: 95.8%          
NT success: 4.2%                 
==== vgg 16 ====                 
Defense accuracy: 86.5%          
NT success: 13.5%                
==== vgg 19 ====                 
Defense accuracy: 84.9%          
NT success: 15.1%                



## Individual test for modified image by FGSM using Inception_v3
==== inception v1 ====         
Defense accuracy: 61.4%        
NT success: 38.6%              
==== inception v2 ====         
Defense accuracy: 66.1%        
NT success: 33.9%              
==== inception v3 ====         
Defense accuracy: 26.6%        
NT success: 73.4%              
==== inception v4 ====         
Defense accuracy: 69.7%        
NT success: 30.3%              
==== inception_resnet_v2 ====  
Defense accuracy: 69.2%        
NT success: 30.8%              
==== resnet_v1_101 ====        
Defense accuracy: 64.9%        
NT success: 35.1%              
==== resnet_v1_152 ====        
Defense accuracy: 66.8%        
NT success: 33.2%              
==== resnet_v2_101 ====        
Defense accuracy: 71.2%        
NT success: 28.8%              
==== resnet_v2_152 ====        
Defense accuracy: 70.8%        
NT success: 29.2%              
==== vgg 16 ====               
Defense accuracy: 55.6%        
NT success: 44.4%              
==== vgg 19 ====               
Defense accuracy: 56.4%        
NT success: 43.6%              


## Predict using Ensemble model for modified image by FGSM using inception_v3
==== ENSEMBLE ====
Defense accuracy: 77.9%
NT success: 22.1%


## Predict using 5 Ensemble model 
### (inception_v4, inception_resnet_v2, resnet_v2_101, resnet_v2_152, vgg19)

