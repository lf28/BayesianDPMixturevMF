function h=logPosteriorKappaPdf(kappa, ss, n, mu_0, C_0, a, b)
%%
% ss: sum_i x_i; a 1*d vector
% n: number of data items
% mu_0, C_0 are prior parameters of mu
% a, b are gamma prior of kappa

d = length(ss);
if kappa < 0
% kappa needs to be positive; imposed by the prior 
    h = log(0);  
else
        
    % assume a gamma prior on kappa; which uses shape and rate parameteration of
    % Gamma 
    logPrior = logGammaPdf(kappa, a, b);

    if n<1
        h = logPrior;
    else
        %h = logPrior + n * logC_d(d, kappa) - logC_d(d, norm(kappa.*ss + C_0.*mu_0)) ;
        h = log(double(kappa>0 & kappa<	Constants.KAPPA_MAX)) +logPrior + n * logC_d(d, kappa) - logC_d(d, norm(kappa.*ss + C_0.*mu_0)) ;
    end
end