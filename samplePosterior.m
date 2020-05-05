function [mu_s,kappa_s] = samplePosterior(uss, hyperpar, ns, sampleMu)
    ss = uss.SS;
    n = uss.nu;
    D=length(ss); %total dimension 
    % sample kappa_s
    f_s = @(x) logPosteriorKappaPdf(x, ss, n, hyperpar.ms_0, hyperpar.Cs_0, hyperpar.a_s, hyperpar.b_s);
    %initial value for slice sampler sets to the ML estimator
    
    try
        if n <= 10
           kappa_s = slicesample(10, ns, 'logpdf', f_s, 'burnin', 10);
        else
            initialK = 5;
            if ~isempty(ss)
            initialK = kappaML(ss, n, D)*0.8;
            end
           kappa_s = slicesample(initialK, ns, 'logpdf', f_s, 'burnin', 10);
        end
    catch ME
        kappa_s = hyperpar.a_s/hyperpar.b_s;
    end

    if sampleMu
        if n > 0
        ss_s = kappa_s.* ss + hyperpar.Cs_0.* hyperpar.ms_0;
        kappa = norm(ss_s);
        mu = ss_s / kappa; 
        mu_s=vsamp(mu', kappa, 1);
        else
            mu_s = vsamp(hyperpar.ms_0', hyperpar.Cs_0, ns);
        end
    else
        mu_s = [];
    end

end
