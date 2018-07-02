function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
lambda_term = lambda / (2 * m);

first_term = (-y .* log(sigmoid(X * theta)));
second_term = (1.-y).*(log(1.-sigmoid(X * theta)));

J = (sum(first_term - second_term) / m) + (lambda_term * sum(theta(2:end) .^ 2));

% === Calculate the gradient of each theta features
difference = sigmoid(X * theta) - y;
% disp(difference);
product = difference .* X;
% disp(product);
theta_sum = sum(product);
% disp(theta_sum);
% disp ((lambda / m) .* theta);
grad = (theta_sum ./ m);
reg_grad = grad(2:end) + transpose((lambda / m) .* theta(2:end));
%disp(reg_grad);
grad = [grad(1);transpose(reg_grad)];
% =============================================================

end
