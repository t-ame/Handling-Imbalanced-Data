% Machine Learning - ECE 6254
% Course Project - Fraud Detection on Imbalanced Sets
% Random Undersampling SVM Isolated
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

%% RANDU - Choose some negative samples, SVM on undersampled dataset
% Here, the value of Ru (ratio of desired nonfraud to fraud)
% can be changed to desired ratio.  This is done by grid search
% The value here "accuracy" is just the standard correct percentage,
% other evaluation techniques addressed in analysis.


% Premanipulation of data
train_ratio_fraud = 0.5;    % Percentage of fraud data to use as training data vs test
train_fraud_X = X(1:floor(N_fraud * train_ratio_fraud),:);
train_fraud_y = y(1:floor(N_fraud * train_ratio_fraud),:);
holdout_fraud_X = X(floor(N_fraud * train_ratio_fraud)+1:N_fraud,:);
holdout_fraud_y = y(floor(N_fraud * train_ratio_fraud)+1:N_fraud,:);

train_ratio_good = 0.5;     % train to holdout percent
% Create undersampled random X values
for N_tests=1:1    % Number of times to run the test (grid search)
    for Ru=[25]  % Ratio of desired nonfraud to fraud, grid search
        rand_vals = sort(randperm(N_good, N_fraud*Ru));
        for i=1:length(rand_vals)
            X_RANDU(i,:) = X(N_fraud + rand_vals(i),:);   %Random undersampled good dataset
        end

        N_RANDU = size(X_RANDU,1);
        train_good_X = X_RANDU(1:floor(N_RANDU * train_ratio_good),:);
        holdout_good_X = X_RANDU(floor(N_RANDU * train_ratio_good)+1:N_RANDU,:);
        train_good_y = zeros(floor(N_RANDU*train_ratio_good),1);
        holdout_good_y = zeros(N_RANDU - floor(N_RANDU*train_ratio_good),1);

        X_train = [train_fraud_X;train_good_X];
        y_train = [train_fraud_y;train_good_y];
        X_test = [holdout_fraud_X;holdout_good_X];
        y_test = [holdout_fraud_y;holdout_good_y];

        RANDU_Model = fitcsvm(X_train,y_train);
        [labels,scores] = predict(RANDU_Model, X_test);
        mislabels = xor(labels,y_test);
        accuracy = 1-sum(double(mislabels))/length(labels);

        fprintf('\nFor ratio of Ru = %f, fraud train-test ratio of %.2f, and good train-test ratio of %.2f, this SVM was %f accurate.\n'...
            , Ru, train_ratio_fraud, train_ratio_good, accuracy*100)
    end
    accuracies(N_tests) = accuracy*100;
end
fprintf('\nAverage accuracy : %f\n',mean(accuracies))
