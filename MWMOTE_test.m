
%WILL NEED TO CONVERT TABLE TO AN ARRAY:
%A = table2array(T)
%DOES IT!!!

%MIGHT NEED TO SPLIT ARRAY:
%lenth = length(xtrain(1,:))
%class1 = xtrain(xtrain(:,end) == 1,1:end-1)
%class0 = xtrain(xtrain(:,end) == 0,1:end-1)
%DOES IT!!!

% s = X(c == 4, :)
%c = cluster no.
%X = data



load fisheriris
rng(1); % For reproducibility
n = size(meas,1);
idx = randsample(n,5);
X = meas(~ismember(1:n,idx),:); % Training data
s = length(X);
Xma = zeros(int32(s/2),length(X(1,:)));
Xmi = zeros(int32(s/2),length(X(1,:)));
ind = s;
for h = 1:(s/2)
    Xma(h,:) = X(h,:);
    Xmi(h,:) = X(ind,:);
    ind = ind - 1;
end

Y = MWMOTE(Xma, Xmi, 4, 5,4,3)




function Xomin = MWMOTE(Xmaj, Xmin, N, k1, k2, k3)
    %[Ximin,Sbmaj] = Imin(Xmaj,Xmin,k1,k2,k3);
    minlen = length(Xmin);
    Xomin = zeros(minlen+N, length(Xmin(1,:)));
    Xomin(1:minlen, :) = Xmin;
    [Sbmaj, Simin, Th] = Imin(Xmaj,Xmin,k1,k2,k3);
    Xomin(minlen+1:end, :) = genSynthetic(Simin,Sbmaj, N, Th, 4); %tune Cp-4
    %Xomin = genSynthetic(Xmin,Xmaj, N);
end

%function to obtain the informed set of minority data points
function [Sbmaj, Simin, AveDist] = Imin(Xmaj,Xmin,k1,k2,k3)
    data = [Xmaj; Xmin];
    dim = length(Xmin(1,:));
    nmin = length(Xmin);
    
    %step 1 - Get all k1 NN for Xmin in the whole data set
    KDT = KDTreeSearcher(data);
    NN = knnsearch(KDT,Xmin, 'K',k1);
    
    %step 2 - populate the Sminf matrix with data only in Xmin
    for i = 1:nmin
        for j = 1:k1
            s = 0;
            for c = 1:nmin
                if Xmin(c,:) == data(NN(i,j),:)
                    s = c;
                end
            end
            if s ~= 0
                if ~exist('Sminf','var')
                  Sminf = data(NN(i,j),:);
                else
                  Sminf = [Sminf; data(NN(i,j),:)];
                end
            end    
        end
    end
    
    %step 3-4 - generate Sbmaj from Sminf
    nminf = length(Sminf);
    KDT = KDTreeSearcher(Xmaj);
    NN = knnsearch(KDT,Sminf, 'K',k2);
    index = 1;
    Sbmaj = zeros(nminf*k2, dim);
    for i = 1:nmin
        for j = 1:k2
            Sbmaj(index,:) = Xmaj(NN(i,j),:);
            index = index + 1;
        end
    end
    
    %step 5-6 - generate Simin using the Xbmaj and Sminf
    nmaj = length(Sbmaj);
    KDT = KDTreeSearcher(Sminf);
    NN = knnsearch(KDT,Sbmaj, 'K',k3);
    
    %Average distance for the synthetic generation...
    [~, D] = knnsearch(KDT,Sminf, 'K', 2, 'IncludeTies',true);
    Ave = 0;
    for i = 1:length(D)
        %D{2}(2)
        Ave = Ave + D{i}(2);
    end
    AveDist = Ave/nminf;
    
    index = 1;
    Simin = zeros(nmaj*k3, dim);
    for i = 1:nmin
        for j = 1:k3
            Simin(index,:) = Sminf(NN(i,j),:);
            index = index + 1;
        end
    end
end


%function to generate the synthetic data points
function Xsyn = genSynthetic(Ximin,Sbmaj, N, Avedist, Cp)
    len = length(Ximin);
    lmaj= length(Sbmaj);
    Sw = zeros(len,1);
    acc = 0;
    
    Xsyn = zeros(N,length(Ximin(1,:)));
    
    %steps 7-9 - Generate weights for each informative minority data point
    for i = 1:len
        for j = 1:lmaj
            acc = acc + InfoWeight(Ximin(i,:),Sbmaj(j,:),Ximin);
        end
        Sw(i) = acc;
        acc = 0;
    end
    s = sum(Sw);
    
    Sp = zeros(len,1);
    for i = 1:len
        Sp(i) = Sw(i)/s;
    end
    
    %steps 10-13 - Generate synthetic data points
    %clustering...
    Th = Avedist * Cp;
    Z = linkage(Ximin,'average','chebychev');
    c = cluster(Z,'cutoff',Th);
    c = categorical(c);
    p = categories(c);
    %cat = length(p);
    for i = 1:length(p)
        %work on each cluster
        %attach the weight to the data
        %clust = [Ximin(c == 'p{i}',:) Sw(c == 'p{i}')];
        j = 0;
        %g = x + (y ? x) × ?
    end
    
    
    
    %Xsyn = Sp;
end


%function for calculating the information weight 
%of each minority data point
function Iw = InfoWeight(x,y,Ximin)
    Iw = cost(x,y) * density(x,y,Ximin);
end

%function to calculate the cost function of a minority data point
function Cf = cost(x,y,Cth,CMAX)
    if ~exist('CMAX','var')
         CMAX = 10;   %choose a suitable value for this
    end
    if ~exist('Cth','var')
         Cth = 3;     %choose a suitable value for this
    end
    dn = dist(x,y);
    fn = 1/dn;
    if fn < Cth
        Cf = (fn/Cth) * CMAX;
    else
        Cf = CMAX;
    end
end


%function to calculate the density function of a minority data point
function Df = density(x,y,Ximin)
    n = length(Ximin);
    cf = cost(x,y);
    sum = cost(Ximin(1,:),y);
    for i = 2:n
        sum = sum + cost(Ximin(i,:),y);
    end
    Df = cf/sum;
end


%normalized euclidean distance function
function dn = dist(x,y)
    n = length(x);
    d = sum((x-y).^2).^0.5;  %Euclidean distance between x and y
    dn = d/n;
end
