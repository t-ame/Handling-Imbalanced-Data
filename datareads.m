% Connor Hudson (chudson43)
% Machine Learning - ECE 6254
% Course Project - Fraud Detection on Imbalanced Sets
%======================================================
%============= Base data reads script =================
%======================================================


Data = xlsread('creditcard.xls');
time = Data(:,1);   % seconds between transaction and first transaction
amt = Data(:,30);           % transaction amount
class = Data(:,31);         % 1 = fraudulent
features = Data(:,2:29);    % features V1-V28 (not including amt)
