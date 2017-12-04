function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X - m x 2, with x values for x0 and x1
% y - column vector of y data
% theta - coulumn vector for theta0 and theta1 respectively
%   X        theta
% [m x 2] * [2 x 1] 
J = (sum(((X * theta) - y) .^ 2)) / (2 * m);

% =========================================================================

end
