function [final_features, final_class] = SMOTE(original_features, original_class, final_ratio)
    ind = find(original_class == 1);

    % P = candidate points
    P = original_features(ind,:);
    T = P';

    % X = Complete Feature Vector
    X = T;

    % Finding the 5 positive nearest neighbours of all the positive blobs
    I = nearestneighbour(T, X, 'NumberOfNeighbours', final_ratio);

    I = I';

    [r c] = size(I);
    S = [];
    th=0.3;
    for i=1:r
        for j=2:c
            index = I(i,j);
            new_P=(1-th).*P(i,:) + th.*P(index,:);
            S = [S;new_P];
        end
    end

    original_features = [original_features;S];
    [r c] = size(S);
    mark = ones(r,1);
    original_class = [original_class;mark];

    train_incl = ones(length(original_class), 1);

    I = nearestneighbour(original_features', original_features', 'NumberOfNeighbours', 4);
    I = I';
    for j = 1:length(original_class)
        len = length(find(original_class(I(j, 2:4)) ~= original_class(j,1)));
        if(len >= 2)
            if(original_class(j,1) == 1)
             train_incl(original_class(I(j, 2:4)) ~= original_class(j,1),1) = 0;
            else
             train_incl(j,1) = 0;   
            end    
        end
    end
    final_features = original_features(train_incl == 1, :);
    final_class = original_class(train_incl == 1, :);

end