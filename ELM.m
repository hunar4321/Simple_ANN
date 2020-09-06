X = [ 0, 0; 1, 1; 0, 1; 1, 0 ];          % Xor data
y = [ 0, 0, 1, 1 ];                      % targets

input = 2; neurons = 5;                  % params.
Wx = randn(input, neurons)*0.01;         % input-hidden weights (range ~ -0.01 to 0.01)
z = tanh(X * Wx);                        % 1st-Layer forward activation (tanh)
Wo = y * pinv(z');                       % Training output weights (closed form solution)
predictions = tanh(X * Wx) * Wo';        % Feedforward propagation - inference

disp(predictions)                        % display the predicted data 
