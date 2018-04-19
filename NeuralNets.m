% Created by Aravind Bala
% Machine Learning - ECE - 6254
% Project - Fraud detection on imbalanced data sets
% Neural network implementation

clc;
clear all;
close all;

% Read Data (Provide input file name here)
input_data = xlsread('TrialData.xlsx');
%input_data = xlsread('creditcard.xlsx');
numlayers = 3;
tansferFcnSelect = 3;                % Set 1 - for log sigmoid, 2 - for tan sigmoid, 3 - for linear transfer function
trainingFcnSelect = 3;               % Set 1 - for gradient descent, 2 - for Scaled Conjugate Gradient, 3 - for variable rate backprop, 
trainingDataSize = 0.5;              % Provide a percent value (0 - 1), to split the training and testing datasets

%% Call SMOTE function

% Separate input and output labels
data_size = size(input_data);
ratio = 3;
[in_data_x, in_data_y] = SMOTE(input_data(:, 1:data_size(2) - 1), input_data(:, data_size(2)), ratio);

% Perform data shuffling
val = [in_data_x, in_data_y];
val = val(randperm(size(val, 1)),:);
in_data_x = val(:, 1:data_size(2) - 1);
in_data_y = val(:, data_size(2));

% Data after smote
data_size_after_smote = size(in_data_x);

%% Input pre-conditioning

trainingdata_size = round(trainingDataSize*data_size_after_smote(1));
%x = input_data(1:trainingdata_size, 1:data_size(2) - 1);
%y = input_data(1:trainingdata_size, data_size(2));

x = in_data_x(1:trainingdata_size, :);
y = in_data_y(1:trainingdata_size, :);

% Normalize the input training data
x_norm  = normalize_data(x);

m = size(x, 1); % Input data size
inputSize = size(x, 2);

%% Create neural network
net = network;

% Set the network parameters
net.numInputs = 1;
net.numLayers = numlayers;
net.biasConnect(:) = 1;
net.inputConnect(1, :) = 1;
for i=2:1:numlayers
    net.layerConnect(i, i-1) = 1;
end
net.outputConnect(numlayers) = 1;

% Set the transfer function
switch tansferFcnSelect
    case 1
        net.layers{:}.transferFcn = 'logsig';
    case 2
        net.layers{:}.transferFcn = 'tansig';
    case 3
        net.layers{:}.transferFcn = 'purelin';
end

% Get the ranges
min_val = min(x_norm, [], 1);
max_val = max(x_norm, [], 1);

% Set the ranges of the input layer
net.inputs{1}.range = [min_val', max_val'];

% Set the neurons for the other layers
for i=1:1:numlayers-1
    net.layers{i}.size = inputSize;
end

% Set the neurons for the output layer
net.layers{numlayers}.size = 1;

% Set the gradient descent backprop network

switch trainingFcnSelect
    case 1
        net.trainFcn = 'traingd';
    case 2
        net.trainFcn = 'trainscg';
    case 3
        net.trainFcn = 'traingdx';
end

net.trainParam.epochs = 10000;
net.trainParam.lr = 0.7;
view(net);

%% Training neural nets
[net, tr] = train(net, x_norm', y');

%% Predict neural net values
% Predict the output values
testdata_size = data_size_after_smote(1) - trainingdata_size;

% Generate the test data using the split percentage
% x_test = input_data(trainingdata_size+1:end, 1:data_size(2) - 1);
% y_test_expected = input_data(trainingdata_size+1:end, data_size(2));

x_test = in_data_x(trainingdata_size+1:end, :);
y_test_expected = in_data_y(trainingdata_size+1:end, :);

% Normalize the test dataset
x_test_norm = normalize_data(x_test);
y_predicted = (return_output(net(x_test_norm'), tansferFcnSelect))';

% Plot confusion matrix
figure;
plotconfusion(y_test_expected', y_predicted');

% Plot the Receiver Operating Characteristics (ROC)
figure;
plotroc(y_test_expected', y_predicted');

%% This function performs normalization of the input data
function data_out = normalize_data(data)
    col_means = mean(data, 1);
    
    % Normalize the input data by the number of dimensions
    col_std = std(data, 1, 1);
    
    % Output the normalized data
    data_out = (data - col_means)./(col_std);
end

%% Function output predictor
function y = return_output(y, tansferFcnSelect)
    switch tansferFcnSelect
        case 1
            for i=1:1:numel(y)
                if(y(i) >= 0.5)
                    y(i) = 1;
                else
                    y(i) = 0;               
                end
            end
            
        otherwise
            for i=1:1:numel(y)
                if(y(i) >= 0)
                    y(i) = 1;
                else
                    y(i) = 0;
                end
            end
    end
end
