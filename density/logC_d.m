function c = logC_d(d, kappa)
%% log normalizing constant of a d-dimensional vMF with kappa 

% if d > 2
%     c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
% else
%     c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-log(besseli(d/2-1,kappa));
% end
c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1, kappa) ; 
end