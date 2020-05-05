function ka = kappaML(ss, n, D)
% helper method for finding ML estimator for kappa 
% ss is the sum of x_i
% n is the sample size 
% D is the dimention of the vMF
    if n == 0
        ka= 10;
        return;
    end
    rbar = norm(ss)/n;
    if (1-rbar) < Constants.EPSILON
        ka=1000;
    else 
        ka=rbar * (D-rbar^2)/(1-rbar^2);
        if ka > Constants.KAPPA_MAX
            ka =Constants.KAPPA_MAX/2;
        end
    end
end