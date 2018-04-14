% Machine Learning - ECE 6254
% Course Project - Fraud Detection on Imbalanced Sets
% SMOTE SVM Isolated
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

%% SMOTE SVM - Pseudo-positive samples, SVM on oversampled dataset
% Premanipulation of data
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

% Create synthetic fraud data using SMOTE and nearest neighbors
data_multiplier = 5;    % Multiple of original fraud size
[train_fraud_X_SMOTE, train_fraud_y_SMOTE] = SMOTE(train_fraud_X, train_fraud_y, data_multiplier);

X_train = [train_fraud_X_SMOTE;train_good_X];
y_train = [train_fraud_y_SMOTE;train_good_y];
X_test = [holdout_fraud_X;holdout_good_X];
y_test = [holdout_fraud_y;holdout_good_y];

SMOTE_Model = fitcsvm(X_train,y_train);
[labels,scores] = predict(SMOTE_Model, X_test);
mislabels = xor(labels,y_test);
accuracy = 1-sum(double(mislabels))/length(labels);
fprintf('SMOTE Model gives standard accuracy of %f percent.\n',accuracy*100)

