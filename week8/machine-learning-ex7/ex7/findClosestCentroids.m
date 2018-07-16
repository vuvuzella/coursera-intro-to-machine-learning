function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X, 1);
for i = 1:m
    dist = 0;
    tempDist = 0;
    centroid_index = 0;
    
    for j = 1:K
        tempDist = vector_distance(X(i,:), centroids(j,:));
        if tempDist < dist || j == 1
            dist = tempDist;
            centroid_index = j;
        end    
    end
    
    idx(i) = centroid_index;
end

% =============================================================

end

function dist = vector_distance(v1, v2)
    diff = v1 - v2;
    dist = sqrt(sum(diff .^ 2));
end