import numpy as np

X=np.asarray([[0,0],[1,1],[0,1],[1,0]]);
y=np.asarray([0,0,1,1]);

inp = 2; hidden = 5; 
Wx = np.random.uniform(-0.01,0.01, (inp, hidden))
z = np.tanh(X @ Wx)
Wo = np.dot(np.linalg.pinv(z), y)
predictions = np.tanh(X @ Wx) @ Wo

print('prediction:', predictions)
