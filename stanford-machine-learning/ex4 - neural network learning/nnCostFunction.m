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

vec_y = zeros(num_labels, m);
for i=1:m,
	vec_y(y(i),i) = 1; % (10 x m)
end;

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

X = [ones(m,1) X];

z2 = Theta1*X'; % (25 x 401) x (401 x 5000) = (25 x 5000) 
a2 = sigmoid(z2); % (25 x 5000)

a2 = [ones(1, m); a2];
z3 = Theta2*a2; % (10 x 26) x (26 x 5000) = (10 x 5000)
a3 = sigmoid(z3); % (10 x 5000)

J = (-1/m)*sum(sum(vec_y.*log(a3)+(1-vec_y).*log(1-a3)));
reg1 = (lambda/(2*m))*(sum(sum((Theta1.^2)(:,2:end)))+sum(sum((Theta2.^2)(:,2:end))));
J += reg1;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for i=1:m,

	a1 = X(i,:)'; 
	z2 = Theta1*a1; % (25 x 401) x (401 x 1) = (25 x 1)
	a2 = sigmoid(z2);

	a2 = [1; a2];
	z3 = Theta2*a2;
	a3 = sigmoid(z3);

	err3 = a3-vec_y(:,i); % (10 x 1)
	err2 = (Theta2')(2:end,:)*err3.*sigmoidGradient(z2); % (25 x 1)

	D1 += err2*a1'; % (25 x 1) x (1 x 401) = (25 x 401)
	D2 += err3*a2'; % (10 x 1) x (1 x 26) = (10 x 26)

end;	

Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

Theta1_grad(:,2:end) += (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m)*Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
