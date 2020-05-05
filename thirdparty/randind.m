function ri = randind(P1,P2)

% RANDIND   Rand Index to Compare Two Partitions
%
%   RI = RANDIND(P1,P2) returns the rand index for partitions
%   P1 and P2 for the same data set. Each of these partitions 
%   are vectors with an index to the group number. For example, 
%   this could be the output from KMEANS or CLUSTER.
%


if length(P1) ~= length(P2)
    error('Input vectors must be the same length.')
    return
end
uP1 = unique(P1);
uP2 = unique(P2);
n1 = length(uP1);
n2 = length(uP2);
n = length(P1);

% Now find the matching matrix M
M = zeros(n1,n2);
I = 0; 
for i = uP1(:)'
    I = I + 1;
    J = 0;
    for j = uP2(:)'
        J = J + 1;
        indI = find(P1 == i);
        indJ = find(P2 == j);
        M(I,J) = length(intersect(indI,indJ));
    end
end
nc2 = nchoosek(n,2);
if n1>1 & n2>1
    % The neither one is a vector, so it is ok to just do the transpose.
    nidot = sum(M);
    njdot = sum(M');
elseif n1==1
    % Then M only has one row. No need to get column totals.
    nidot = M;
    njdot = sum(M);
else
    % Then M has one column. No need to get row totals.
    nidot = sum(M);
    njdot = M;
end
    
ntot = sum(sum(M.^2));
num = nc2 + ntot - 0.5*sum(nidot.^2) - 0.5*sum(njdot.^2);
ri = num/nc2;


