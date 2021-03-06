function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
regTheta = (lambda/(2*m))*sum(theta(2:end).^2);

J = (1/m)*((-y).'*log(h)-(1-y).'*log(1-h))+regTheta;
theta(1) = 0;
grad = (1/m)*(X.'*(h-y)+lambda*theta);

grad = grad(:);
end