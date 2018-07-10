function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% NN Cost function without regularization
a1 = [ones(m,1) X];
% disp(size(Theta1));

z2 = Theta1 * transpose(a1);
a2 = sigmoid(z2);
m_a2 = size(a2, 2);
a2 = [ones(1, m_a2); a2];
% disp(size(a2));

z3 = Theta2 * a2;
a3 = transpose(sigmoid(z3));
% disp(size(a3));
% disp(size(y));

cost_term = zeros(size(a3));
% first_term = zeros(size(a3, 2));  % preallocation not needed
% second_term = zeros(size(a3, 2)); % preallocation not needed
% disp(size(first_term));
% disp(size(a3(1, :)));

% create augmented label
output = 1:num_labels;
aug_label = gen_aug_labels(y,output');

for n = 1:m
    % aug_label = zeros(1, num_labels);
    % aug_label(y(n)) = 1;
    
    % disp(y(n));
    % disp(aug_label);
    
    first_term = -aug_label(n, :) .* log(a3(n,:));
    second_term = (1 - aug_label(n, :)) .* log(1 - a3(n, :));
%    disp(size(first_term));
%    disp(size(second_term));

    cost_term(n, :) = first_term - second_term;
end

% disp(sum(sum(cost_term, 2), 1)/m);

sum_K = sum(cost_term, 2);
sum_m = sum(sum_K);

J = sum_m / m;

% Calculate regularization term

% Calculate Theta1 regularization
theta1_row_size = size(Theta1, 1);
theta1_sum_features = zeros(theta1_row_size, 1);
% disp(theta1_row_size)
for n = 1:theta1_row_size
    theta1_sum_features(n, :) = sum(Theta1(n, 2:end) .^ 2, 2);
end
% disp(size(theta1_sum_features));
theta1_sum_m = sum(theta1_sum_features);

% Calculate Theta2 regularization
theta2_row_size = size(Theta2, 1);
theta2_sum_features = zeros(theta2_row_size, 1);
for n = 1:theta2_row_size
    theta2_sum_features(n, :) = sum(Theta2(n, 2:end) .^ 2, 2);
end
% disp(size(theta2_sum_features));
theta2_sum_m = sum(theta2_sum_features);

% Calculate total regularization term
reg_term = (lambda / (2 * m))  * (theta1_sum_m + theta2_sum_m);

% Calulate regularized cost / loss function value
J = J + reg_term;   % Forward Propagation with regularization

theta1_accum = zeros(size(Theta1));
theta2_accum = zeros(size(Theta2));



% Back propagation
for n = 1:m

    % feed forward
    f_a1 = [1;transpose(X(n,:))]; % 401 x 1
    % disp(size(a1));
    f_z2 =Theta1 * f_a1;   % 25 x 1
    f_a2 = [1; sigmoid(f_z2)];  % 26 x 1
    % disp(size(f_a2));
    f_z3 = Theta2 * f_a2;
    % disp(size(f_z3));
    f_a3 = sigmoid(f_z3);   % 10 x 1 h_x
    % disp(size(f_a3));
    
    % Feed Backward
    % delta 3 for output layer
    % disp(f_a3);
    d3 = f_a3 - transpose(aug_label(n, :)); % 10 x 1
    % disp(size(d3));
    
    d2 = (transpose(Theta2(:,2:end)) * d3) .* sigmoidGradient(f_z2); % 26 x 1
    % d2 = d2(2:end); % 25 x 1 bias term removed
    % disp(size(d2));
    
    theta1_accum = theta1_accum + (d2 * transpose(f_a1));
    theta2_accum = theta2_accum + (d3 * transpose(f_a2));
end

Theta1_grad = theta1_accum / m;
Theta2_grad = theta2_accum / m;

% Regularize the gradient
theta1_reg_term = (lambda / m) * Theta1(:,2:end);
theta1_reg_term = [zeros(size(Theta1,1), 1) theta1_reg_term];
theta2_reg_term = (lambda / m) * Theta2(:,2:end);
theta2_reg_term = [zeros(size(Theta2,1), 1) theta2_reg_term];

Theta1_grad = Theta1_grad + theta1_reg_term;
Theta2_grad = Theta2_grad + theta2_reg_term;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function aug = gen_aug_labels(labels, K)
    h = size(labels, 1);
    aug = zeros(size(labels, 1), size(K, 1));

    for i = 1:h
        aug(i,:) = K;
    end
    aug = aug == labels;
end