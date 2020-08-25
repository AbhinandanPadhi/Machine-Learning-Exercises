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

X1 = [ones(m, 1) X];
A2 = sigmoid(X1 * Theta1');
A2 = [ones(m, 1) A2];
A3 = sigmoid(A2 * Theta2');
% [~, p] = max(A3, [], 2);
% disp(A3);

y1 = zeros(m, num_labels);

for i = 1:m
    temp = y(i);
    y1(i, temp) = 1;
end

for k = 1:num_labels
    A = A3(:, k);
    B = log(A);
    C = B' * y1(:, k);
    D = log(1 - A);
    E = 1 - y1(:, k);
    F = D' * E;
    J = J + (-1 / m) * (C + F);
end

T1 = 0;
T2 = 0;

for j = 1:hidden_layer_size
    for k = 1:input_layer_size
        T1 = T1 + (Theta1(j, k + 1) ^ 2);
    end
end

for j = 1:num_labels
    for k = 1:hidden_layer_size
        T2 = T2 + (Theta2(j, k + 1) ^ 2);
    end
end


R = T1 + T2;
J = J + (lambda / (2 * m)) * R;

acc1 = zeros(size(Theta1));
acc2 = zeros(size(Theta2));

for t = 1:m
    a1 = X(t, :);
    a1 = a1';
    a1 = [1; a1];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    d3 = a3 - (y1(t, :))';
    d2 = (Theta2' * d3) .* [1; sigmoidGradient(z2)];
    d2 = d2(2:end);
    
    acc1 = acc1 + d2 * a1';
    acc2 = acc2 + d3 * a2';
end

D1 = (1 / m) * acc1;
D2 = (1 / m) * acc2;

tempTheta1 = Theta1(:, 2:end);
tempTheta2 = Theta2(:, 2:end);

D1(:, 2:end) = D1(:, 2:end) + (lambda / m) * tempTheta1;
D2(:, 2:end) = D2(:, 2:end) + (lambda / m) * tempTheta2;

Theta1_grad = D1;
Theta2_grad = D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
