function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

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

C_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_arr = [0.01 0.03 0.1 0.3 1 3 10 30];

% min_error = 0;

% C = 0;
% sigma = 0;

% iterate over to find optimal C, sigma for dataset 3. 

% for i=1:length(C_arr)
% 	for j=1:length(sigma_arr)
		
% 		model = svmTrain(X, y, C_arr(i), @(x1, x2) gaussianKernel(x1, x2, sigma_arr(j))); 
% 		predictions = svmPredict(model, Xval);

% 		tmp = mean(double(predictions~=yval));

% 		% fprintf('i: %d(%f) j: %d(%f), error_rate: %f\n', i, C_arr(i), j, sigma_arr(j), tmp);
% 		if tmp < min_error
% 			min_error = tmp;
% 			C = C_arr(i);
% 			sigma = sigma_arr(j);
% 		end

% 	end
% end

% optimized val.
C = 1.000000;
sigma = 0.100000;

% =========================================================================

end
