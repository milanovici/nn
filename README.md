# Start command:  
```python bn.py --weight-init xavier --bias-init zero --batch-norm True --dropout False --layers 2 --activation relu --training-epochs 5 --learning-rate 0.01 ```

# Possible values of params :  
weight-init [xavier, he]  
bias-init [zero, normal]  
batch-norm [True, False]  
dropout [True, False]  
layers [1,2,3]  
activation [relu, relu6, sigmoid, elu, softplus]  
training-epochs int  
learning-rate float  


# Versions:  
Python 3.6.1 :: Anaconda 4.4.0 (64-bit)    
tensorflow==1.3.0    
