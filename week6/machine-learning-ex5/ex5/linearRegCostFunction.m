function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate Regularized Cost/loss function
% Cost term
h_x = X * theta;
calc_actual_diff = (h_x - y) .^ 2;
cost_term = sum(calc_actual_diff) / (2 * m);

% Regularization term
reg_term = (lambda / (2 * m)) * (sum(theta(2:end).^2));

J = cost_term + reg_term;

% Calculate the gradient at these theta values
% disp(size(X));
% disp(size(theta));
theta_0 = sum((h_x - y) .* X(:,1)) / m;
% disp(size(theta_0));
grad_reg_term = (lambda / m) .* theta(2:end);
% disp(size(grad_reg_term));

% partially vectorized.
% theta_j = zeros(size(X, 2) - 1, 1);
% for i = 2:size(X, 2)
%     theta_j(i-1) = (sum((h_x - y) .* X(:, i)) / m) + grad_reg_term(i-1);
% end

% Vectorized calculation
theta_j = (((h_x - y)' * X(:, 2:end)) / m) + grad_reg_term';
% disp(size(theta_j));

grad = [theta_0;theta_j(:)];
% disp(size(grad));



% =========================================================================

grad = grad(:);

end
