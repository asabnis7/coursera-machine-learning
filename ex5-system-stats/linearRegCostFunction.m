function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Cost Function + Regularization
predict = X*theta;
sqError = (predict - y).^2;
J = (1/(2*m))*sum(sqError) + (lambda/(2*m))*sum(theta(2:end).^2);

% Gradient Descent
grad = (1/m)*(X'*(X*theta-y));
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);
grad = grad(:);

end