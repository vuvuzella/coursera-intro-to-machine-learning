function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add the bias theta on all inputs
aug_inputs = [ones(m, 1) X];
% disp(size(aug_inputs));
% disp(size(Theta1));

% Predict using the hypothesis function
% and the trained theta -> hidden layer 1
hidden_1 = sigmoid(aug_inputs * transpose(Theta1));
% disp(size(hidden_1));

% Add in the bias theta in the hidden_1's outputs
bias_term = ones(size(hidden_1, 1), 1);
% disp(size(bias_term));

aug_hidden_1 = [bias_term hidden_1];
% disp(size(aug_hidden_1));
output = sigmoid(aug_hidden_1 * transpose(Theta2));

[M, p] = max(output, [], 2);




% =========================================================================


end
