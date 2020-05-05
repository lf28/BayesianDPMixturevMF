function [z_sample, mcmc_mu, mcmc_kappa] = MvMF_inference(Y, niter, KK)
% Y: the input dataset
% niter: number of iterations  
% KK: mixture size
% This function implements a Gibbs sampler for finite mixture of von Mises
% Fisher distribution where only the proportion parameter $pi$ is
% integrated out 
% the state of the chain include z: n*1, mu: p*KK, and kappa: KK*1
% Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk;
% distributable under GPL, see README.txt

[N, D]=size(Y);
ss_y = sum(Y, 1);
hyperp.ms_0=ss_y/norm(ss_y); 
hyperp.Cs_0=0.01; 

hyperp.a_s=1;
hyperp.b_s=0.001;

z_sample = zeros(N, niter);

mcmc_mu = zeros(KK,D,niter);
mcmc_kappa = zeros(KK, niter);
% N is the number of data items
% P is the total dimension   
%% Initialise the state of the chain
% initialize the membership z_i for i =1,...,N
%z = kmeans(Y, KK);
z = randi(KK, N, 1);
z_sample(:,1) =z;
m = zeros(1,KK);
pars = containers.Map('KeyType','int32','ValueType','any');
initKappas = ones(1, KK);
for j = 1:KK
   idx = find(z==j);
   m(j) = length(idx);
   ss = sum(Y(idx,:), 1); 
   nj = length(idx);
   [mu_s,  kappa_s]=samplePosterior(ss, nj, hyperp, initKappas(j));
   mcmc_mu(j,:,1) = mu_s; 
   mcmc_kappa(j, 1) = kappa_s;
   pars(j) = struct('mus', mu_s,'kappas', kappa_s);
end

%% Gibbs steps start 
for t = 2: niter
    
    % update class membership
    % the proportion parameter is integrated out 
    if KK > 1
        for i = 1:N
             m(z(i)) = m(z(i)) - 1;
             z(i) = sample_z(Y(i,:), pars, m, KK);
             m(z(i)) = m(z(i)) + 1;
        end
    end
    
    z_sample(:, t) = z;

    %update component parameters
    ind = find(m);
    for j=1:length(ind)
        idx = find(z==ind(j));
        ss = sum(Y(idx,:), 1);  
        n_ = length(idx);
        %if n_ >1 % can skip this update step if n_ =1 ;
            [tmpmu_s,  tmpkappa_s]=samplePosterior(ss, n_, hyperp, pars(ind(j)).kappas);
            mcmc_mu(j,:,t) = tmpmu_s; 
            mcmc_kappa(j, t) = tmpkappa_s;
            pars(ind(j)) = struct('mus', tmpmu_s,  'kappas', tmpkappa_s);
        %end
    end
    
    %optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
    %k = length(unique(z));
    
    if mod(t, 10) ==0
       fprintf("Iteration: %d.\n", t); 
    end
end



end


function kk = sample_z(data, pars_, m, KK)
    D=length(data);
    mu_s = zeros(KK, D);
    kappa_s = zeros(1,KK);
    for i_ = 1: KK
        mu_s(i_,:) = pars_(i_).mus;       
        kappa_s(i_) = pars_(i_).kappas;
    end
    logPost = log(m)' + logVmf(data', mu_s', kappa_s) ; 
    logP = logPost';
    pp = exp(logP - max(logP)); % -max(p) for numerical stability
    pp = pp / sum(pp);
    u=rand(1);
    ind = find(cumsum(pp(1:(length(pp))))>=u, 1 );        
    kk = ind;
end


function [mu_s,kappa_s] = samplePosterior(ss, n, hyperpar, oldkappa)
    D=length(ss); %total dimension 
    % sample kappa_s
    f_s = @(x) logPosteriorKappaPdf(x, ss, n, hyperpar.ms_0, hyperpar.Cs_0, hyperpar.a_s, hyperpar.b_s);
    try
        kappa_s = slicesample(oldkappa, 1, 'logpdf', f_s, 'burnin', 5);
    catch ME
        kappa_s = kappaML(ss, n, D)/2;
    end
    if n > 0
        ss_s = kappa_s.* ss + hyperpar.Cs_0.* hyperpar.ms_0;
        kappa = norm(ss_s);
        mu = ss_s / kappa; 
        mu_s = vsamp(mu', kappa, 1);
    else
        mu_s = vsamp(hyperpar.ms_0', hyperpar.Cs_0, 1);
    end
end



