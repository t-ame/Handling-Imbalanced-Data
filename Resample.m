%% Load data, split into classes

creditcard = struct2array(load('creditcarddata.mat'));
creditsize = size(creditcard);

class0 = creditcard(creditcard(:,end) == 0,1:end-1);
class1 = creditcard(creditcard(:,end) == 1,1:end-1);

testratio = 0.3;    %ratio of data to be used for testing
c1= round(testratio*length(class1));
c0= round(testratio*length(class0));
%% extract and save test data

testdata = zeros(c1+c0, creditsize(2));
testdata(1:c0, 1:end-1) = class0(1:c0, :);
testdata(c0+1:end, 1:end-1) = class1(1:c1, :);
testdata(c0+1:end, end) = testdata(c0+1:end, end)+1;
save('testData.mat', 'testdata');

%% remove test data from class

class0 = class0(c0+1:end, :);
class1 = class1(c1+1:end, :);

%% MWMOTE resampling

N = round(0.5*length(class0));
k1 = 15;
k2 = 13;
k3 = 10;

ResampledClass1 = MWMOTE(class0, class1, N, k1, k2, k3, 5);

%% Create output variable
%if exist('class1.mat', 'file') ~= 2 
    save('class1.mat', 'ResampledClass1');
%end

%% Create output variable
%if exist('class0.mat', 'file') ~= 2 
    save('class0.mat', 'class0');
%end

%% Clear temporary variables
%clearvars testdata filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp R;
