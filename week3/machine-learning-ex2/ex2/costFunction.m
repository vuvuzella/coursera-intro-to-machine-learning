function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% === Calculate the cost functio === 
% Calculate the first term of the logistic regression cost function
first_term = -y .* log(sigmoid(X * theta));
% disp(first_term);
% Calculate the second term of the logistic regression cost function
second_term = (1.-y).*(log(1.-sigmoid(X * theta)));
% disp(second_term);
% Calculate the sum of the difference of the first and second term vectors
sum_total = sum(first_term - second_term);
% Calculate the average cost with the given theta
J = sum_total / m;

% === Calculate the gradient of each theta features
difference = sigmoid(X * theta) - y;
% disp(difference);
product = difference .* X;
% disp(product);
theta_sum = sum(product);
% disp(theta_sum);
grad = theta_sum ./ m;

% =============================================================

end
