function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
c_vec_length = length(c_vec);
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec_length = length(sigma_vec);
models = [];

for i = 1:c_vec_length
    for j = 1:sigma_vec_length
        newmod = svmTrain(X, y, c_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        pred = svmPredict(newmod, Xval);
        pred_error = mean(double(pred ~= yval));
        newStruct = struct('c', c_vec(i), 'sigma', sigma_vec(j), 'model', newmod, 'pred_error', pred_error);
        models = [models; newStruct];
    end
end

% disp(size(models));
% disp(models);
models_count = length(models);

optimum_c_sigma = models(1);

for i = 1:models_count
    if models(i).pred_error < optimum_c_sigma.pred_error
        optimum_c_sigma = models(i);
    end
end

% disp('optimum C: ');
% disp(optimum_c_sigma.c);
% disp('optimum sigma: ');
% disp(optimum_c_sigma.sigma)

c = optimum_c_sigma.c;
sigma = optimum_c_sigma.sigma;





% =========================================================================

end
