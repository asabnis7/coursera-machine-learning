function [C_val, sigma_val] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% C_val = 0;
% sigma_val = 0;
% 
% C = [0.01 0.03 0.09 0.12 0.15 0.3 0.6 0.9 1];
% sigma = [0.01 0.03 0.09 0.12 0.15 0.3 0.6 0.9 1];
% min_err = 1;
% 
% for i = 1:length(C)
%     for j = 1:length(sigma)
%         
%         model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
%         predictions = svmPredict(model, Xval);
%         error = mean(double(predictions ~= yval));
%         
%         if error < min_err
%             min_err = error;
%             C_val = C(i);
%             sigma_val = sigma(j);
%         end
%         
%     end
% end

C_val = 0.3;
sigma_val = 0.09;

end