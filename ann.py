import numpy as np
X=np.array[[0,0],[0,1],[1,0],[1,1]]
y=np.array[[0],[1],[1],[0]]
np.random.seed(1)
weights=np.random.rand(2,1)
for i in range(10000):
    output=1/(1+np.exp(-np.dot(X,weights)))
    error=output
    weights+=np.dot(X.T,error*output*(1-output))
    print("Output after tarining data")
    print(output)
