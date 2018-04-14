% Machine Learning - ECE 6254
% Course Project - Fraud Detection on Imbalanced Sets
% Weighted SVM Isolated Model Analysis
%======================================================

clear
clc
close all

%Data = xlsread('creditcard.xls');    Initial loads
%save('Data.mat','Data')
load('Data.mat')
Data_sorted = sortrows(Data,31,'descend');  %1:169 = bad
time = Data_sorted(:,1);   % seconds between transaction and first transaction
amt = Data_sorted(:,30);           % transaction amount
class = Data_sorted(:,31);         % 1 = fraudulent
features = Data_sorted(:,2:29);    % features V1-V28 (not including amt)
N = length(class);
N_fraud = sum(class(:)==1);
N_good = sum(class(:)==0);

X = [time,features,amt];
y = class;

%% Weighted SVM - Cost manipulation
% This model can be altered using the cost_matrix variable below.
% This is defined as in the fitcsvm notes, with Cost(i,j) being the
% cost of classifying point i into class j

train_ratio_fraud = 0.5;    % Percentage of fraud data to use as training data vs test
train_fraud_X = X(1:floor(N_fraud * train_ratio_fraud),:);
train_fraud_y = y(1:floor(N_fraud * train_ratio_fraud),:);
holdout_fraud_X = X(floor(N_fraud * train_ratio_fraud)+1:N_fraud,:);
holdout_fraud_y = y(floor(N_fraud * train_ratio_fraud)+1:N_fraud,:);

train_ratio_good = 0.5;     % Train versus holdout data percent
train_good_X = X(N_fraud+1:N_fraud+1+floor(N_good * train_ratio_good),:);
train_good_y = y(N_fraud+1:N_fraud+1+floor(N_good * train_ratio_good),:);
holdout_good_X = X(N_fraud+floor(N_good * train_ratio_good):end,:);
holdout_good_y = y(N_fraud+floor(N_good * train_ratio_good):end,:);

X_train = [train_fraud_X;train_good_X];
y_train = [train_fraud_y;train_good_y];
X_test = [holdout_fraud_X;holdout_good_X];
y_test = [holdout_fraud_y;holdout_good_y];
        
cost_matrix = [0,1;2,0];   %Cost(i,j) = cost of classifying point i into class j

Weighted_Model = fitcsvm(X_train,y_train);
[labels,scores] = predict(Weighted_Model, X_test);
mislabels = xor(labels,y_test);
accuracy = 1-sum(double(mislabels))/length(labels);
fprintf('Weighted Model gives standard accuracy of %f percent.\n',accuracy*100)

