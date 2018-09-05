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

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1), X];
z2 = a1*Theta1';
a2 = [ones(m,1), sigmoid(z2)];
z3 = a2*Theta2';
hx = sigmoid(z3);

yk = zeros(m,num_labels);
for l = 1:m
    yk(l,y(l)) = 1;
end

for i = 1:m
    for k = 1:num_labels
        J = J + yk(i,k)*log(hx(i,k)) + (1-yk(i,k))*log(1-hx(i,k));
    end
end
J = -J/m;

% Adding regularization
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Delta_1 = 0; Delta_2 = 0;

% for t = 1:m
delta3 = hx - yk;
delta2 = (delta3*Theta2).*sigmoidGradient([zeros(size(z2,1),1), z2]);
delta2 = delta2(:,2:end);

Delta_1 = (delta2'*a1);
Delta_2 = (delta3'*a2);
% end

Theta1_grad = Delta_1./m + (lambda/m).*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Delta_2./m + (lambda/m).*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end