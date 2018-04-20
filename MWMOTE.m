%reference = http://www.cs.bham.ac.uk/~xin/papers/tkde2012_IslamYao.pdf


function Xomin = MWMOTE(Xmaj, Xmin, N, k1, k2, k3, clustCost)
    minlen = length(Xmin);
    Xomin = zeros(minlen+N, length(Xmin(1,:)));
    Xomin(1:minlen, :) = Xmin;
    [Sbmaj, Simin, Avedist, Maxdist] = Imin(Xmaj,Xmin,k1,k2,k3);
    Threshold = Avedist*clustCost;   %tune Cp = 4
    SDmin = std2(Xmin);
    Xomin(minlen+1:end, :) = genSynthetic(Simin,Sbmaj, N, Threshold,SDmin,Maxdist);
end


%function to obtain the informed set of minority data points
function [Sbmaj, Simin, AveDist, maxDist] = Imin(Xmaj,Xmin,k1,k2,k3)
    data = [Xmaj; Xmin];
    dim = length(Xmin(1,:));
    nmin = length(Xmin);
    
    %step 1 - Get all k1 NN for Xmin in the whole data set
    'Step 1'
    KDT = KDTreeSearcher(data);
    NN = knnsearch(KDT,Xmin, 'K',k1);
    
    %step 2 - populate the Sminf matrix with data only in Xmin
    'Step 2'
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
    
    Sminf = removeDuplicate(Sminf);
    
    %step 3-4 - generate Sbmaj from Sminf
    'Step 3-4'
    nminf = length(Sminf);
    KDT = KDTreeSearcher(Xmaj);
    [NN, De] = knnsearch(KDT,Sminf, 'K',k2);
    
    Ave = 0;
    for i = 1:length(De)
        Ave = Ave + De(i,k2);
        %Ave = Ave + De{i}(k2);
    end
    maxDist = Ave/nminf;
    
    index = 1;
    Sbmaj = zeros(nminf*k2, dim);
    for i = 1:nminf
        for j = 1:k2
            Sbmaj(index,:) = Xmaj(NN(i,j),:);
            index = index + 1;
        end
    end
    
    Sbmaj = removeDuplicate(Sbmaj);
    
    %step 5-6 - generate Simin using the Xbmaj and Sminf
    'Step 5-6'
    nmaj = length(Sbmaj);
    KDT = KDTreeSearcher(Sminf);
    NN = knnsearch(KDT,Sbmaj, 'K',k3);
    
    %Average distance for the synthetic generation...
    [~, D] = knnsearch(KDT,Sminf, 'K', 2, 'IncludeTies',true);
    Ave = 0;
    for i = 1:length(D)
        Ave = Ave + D{i}(2);
    end
    AveDist = Ave/nminf;
    
    index = 1;
    Simin = zeros(nmaj*k3, dim);
    for i = 1:nmaj
        for j = 1:k3
            Simin(index,:) = Sminf(NN(i,j),:);
            index = index + 1;
        end
    end
    Simin = removeDuplicate(Simin);
end


%function to generate the synthetic data points
function Xsyn = genSynthetic(Ximin,Sbmaj, N, Th, Cth, CMAX)
    len = length(Ximin);
    lmaj= length(Sbmaj);
    Sp = zeros(len,1);
    %acc = 0;
    
    %steps 7-9 - Generate weights for each informative minority data point
    'Step 7-9'
    for i = 1:len
        for j = 1:lmaj
            Sp(i) = Sp(i) + (cost(Ximin(i,:),Sbmaj(j,:),Cth, CMAX) * density(Ximin(i,:),Sbmaj(j,:),Ximin));
        end
        %Sp(i) = acc;
        %acc = 0;
    end
    s = sum(Sp);
    Sp = Sp/s;
    
    %Sp = zeros(len,1);
    %for i = 1:len
     %   Sp(i) = Sp(i)/s;
    %end
    
    %steps 10-13 - Generate synthetic data points
    'Step 10-13'
    %clustering...
    Z = linkage(Ximin,'average');
    c = cluster(Z,'cutoff',Th);
    
    Xsyn = zeros(N,length(Ximin(1,:)));
    data = [Ximin c];
    
    Ngen = datasample(data, N, 1, 'Weights', Sp);
    
    for i = 1:N
        clust = data(data(:,end) == Ngen(i,end),1:end-1);
        KDT = KDTreeSearcher(clust);
        NN = knnsearch(KDT,Ngen(i,1:end-1), 'K',randi([2 4],1,1));
        Xsyn(i,:) = Ngen(i,1:end-1) + (clust(NN(end),:) - Ngen(i,1:end-1)) * rand;
    end
end



%function to calculate the cost function of a minority data point
function Cf = cost(x,y,Cth,CMAX)
     %'    Cost function'
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
    %'     Density function'
    n = length(Ximin);
    cf = cost(x,y);
    sum = cost(Ximin(1,:),y);
    for i = 2:n
        sum = sum + cost(Ximin(i,:),y);
    end
    Df = cf/sum;
end


%function to remove duplicates in generated matrix... for length management
function output = removeDuplicate(X)
    lenx = length(X);
    dim = length(X(1,:));
    logs = zeros(lenx,1);
    
    for i = 1:lenx
        if logs(i) ~= 1
            for j = 1:lenx
                if (j ~= i) && (logs(j) ~= 1)
                    s=0;
                    for d = 1:dim
                        s = s + abs(X(i,d) - X(j,d));
                    end
                    if s == 0
                        logs(j) = 1;
                    end
                end
            end
        end
    end

    output = X(logs == 0,:);
end

%normalized euclidean distance function
function dn = dist(x,y)
    n = length(x);
    d = sum((x-y).^2).^0.5;  %Euclidean distance between x and y
    dn = d/n;
end
