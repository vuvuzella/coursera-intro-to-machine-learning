function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
% sX_norm = X;                     % m x 2 matrix
% mu = zeros(1, size(X, 2));      % 1 x 2 row vector
% sigma = zeros(1, size(X, 2));   % 1 x 2 row vector

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% row_size = size(X, 1);
% 
% size_mean = zeros(row_size, 1);
% size_mu = mean(X(:,2));
% size_mean(:) = size_mu;
% size_diff = X(:,2) - size_mean;
% 
% size_std = zeros(row_size, 1);
% size_sigma = std(X(:,2));
% size_std(:) = size_sigma;
% 
% size_norm = size_diff ./ size_std;
% 
% bed_mean = zeros(row_size, 1);
% bed_mu = mean(X(:,2));
% bed_mean(:) = bed_mu;
% bed_diff = X(:,2) - bed_mean;
% 
% 
% bed_std = zeros(row_size, 1);
% bed_sigma = std(X(:,2));
% bed_std(:) = bed_sigma;
% 
% bed_norm = bed_diff ./ bed_std;
% 
% X_norm(:,1) = size_norm;
% X_norm(:,2) = bed_norm;
% mu(:,1) = size_mu;
% mu(:,2) = bed_mu;
% sigma(:,1) = size_sigma;
% sigma(:,2) = bed_sigma;

mu = mean(X); % returns mean of size, # of bedrooms, and price
sigma = std(X); % returns the standard deviation of size, bedrooms and price

mu_matrix = ones(size(X,1), 1) * mu;
sigma_matrix = ones(size(X, 1), 1) * sigma;

X_norm = (X - mu_matrix) ./ sigma_matrix;


% ============================================================

end

