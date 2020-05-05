function [z_sample, logllh] = iMvMF_inference(Y, niter, calLogllh, mcN, updateHyperPrior, initialK, hyperPar)

% the state of the chain include z and phi

% Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk;
% distributable under GPL, see README.txt
if (~exist('mcN', 'var'))
   mc_n =  30;
else
   mc_n = mcN;
end


if (~exist('updateHyperPrior', 'var'))
   updateHyperPrior = false;
else
   %updateHyperPrior = false;
end


[N, D]=size(Y);
ss_y = sum(Y, 1);

if (~exist('hyperPar', 'var'))
hyperp.a_s=3;
hyperp.b_s=0.1;
% hyperp.a_t=3;
% hyperp.b_t=0.1;   
else
hyperp = hyperPar;
end

hyperp.alpha = 1;
hyperp.ms_0=ss_y/norm(ss_y); 
hyperp.Cs_0=1; 
%hyperp.mt_0=ss_y((D-1):(D))/norm(ss_y((D-1):(D))); 
%hyperp.Ct_0= 0.5;
%rbar_s = norm(ss_y(1:(D-2)))/N;
%rbar_t = norm(ss_y((D-1):(D)))/N;
%kappaML_s = rbar_s * ((D-2)-rbar_s^2)/(1-rbar_s^2);
%kappaML_t = rbar_t * (2-rbar_t^2)/(1-rbar_t^2);

% hyperp.a_s=3;
% hyperp.b_s=0.1;
%hyperp.a_t=1;
%hyperp.b_t=1/5;



if (~exist('initialK', 'var'))
   initialK =  20;
else
 
end

kappa_s_ml = kappaML(sum(Y,1), N, D-2);

uss_0.nu = 0;
uss_0.SS = zeros(1,D);
logllh = zeros(1, niter);

priorSampleSize = 1000;
[mu_prior_sample,  kappa_prior_sample]=samplePosterior(uss_0, hyperp, priorSampleSize, true);

z_sample = zeros(N, niter);
% N is the number of data items
% P is the total dimension   
%% Initialise the state of the chain

% initialize the membership z_i for i =1,...,N
%initialK = 4;
if initialK > N
   initialK = N; 
end
z = randi(initialK, N, 1);
%z = kmeans(Y, initialK);
z_sample(:,1) =z;
m = zeros(1,N);
%pars = containers.Map('KeyType','int32','ValueType','any');
% sample phi
Ks = unique(z);
K = length(Ks); 
%Ms = zeros(K, D);
%Kappas = zeros(K, 2);
initial_diffused_kappa_s = Constants.INIT_KAPPA;

if initial_diffused_kappa_s > kappa_s_ml
    initial_diffused_kappa_s = kappa_s_ml;
end

U_mu = zeros(N,D); %**
U_kappa = zeros(1,N); %**
U_SS = struct('nu', cell(N,1), 'SS', cell(N,1)); %**
for j = 1:K
   idx = find(z==Ks(j));
   m(Ks(j)) = length(idx);
   ss = sum(Y(idx,:), 1); 
   nj = length(idx);
   U_SS(Ks(j)).nu = nj;
   U_SS(Ks(j)).SS = ss;
   [mu_s,  kappa_s]=samplePosterior(U_SS(j), hyperp, 1, true);
   %pars(Ks(j)) = struct('mus', mu_s,'kappas', kappa_s);
   U_mu(Ks(j),:) = mu_s;
   U_kappa(Ks(j))= initial_diffused_kappa_s; 
end

%optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
if calLogllh
ind = find(m);
ss_mat=cell2mat({U_SS(ind).SS}');
logllh(1) =calLoglik();
end

%% Gibbs steps start 

%mc_n= 10;
for t = 2: niter
    loglik = 0;
    %update class membership
    for i = 1:N
         m(z(i)) = m(z(i)) - 1;
         U_SS(z(i)) = downdate_SS(Y(i,:), U_SS(z(i)));

         [z(i), ll_] = sample_z(Y(i,:), m, U_mu, U_kappa, hyperp, mc_n);
         loglik=loglik+ll_;
         m(z(i)) = m(z(i)) + 1;
         
         
         if m(z(i))>1
            U_SS(z(i)) = update_SS(Y(i,:), U_SS(z(i)));
         else
            U_SS(z(i)) = update_SS(Y(i,:), uss_0);
            [U_mu(z(i),:),  U_kappa(z(i))] = samplePosterior(U_SS(z(i)), hyperp, 1, true);
          end       
    end
    
    z_sample(:, t) = z;

    %update component parameters

    ind = find(m);
    
    %tmpMus = zeros(length(ind), D);
    for j=1:length(ind)
        %if n_ >1 % can skip this update step if n_ =1 ;
            [U_mu(ind(j),:),  U_kappa(ind(j))]=samplePosterior(U_SS(ind(j)), hyperp, 1, true);
        %end
    end
    
    ss_mat=cell2mat({U_SS(ind).SS}');
    if calLogllh
        %logllh(t) =loglik + calDPLogLik(); 
        logllh(t) =calLoglik();
    end
    %optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
    
    %updateHyperPrior = false;
    
    if updateHyperPrior 
    ms_0 = sum(normr(ss_mat), 1);
     ms_0 = ms_0/norm(ms_0);
%    ms_0 = sum(tmpMus,1);
     hyperp.ms_0 = ms_0/norm(ms_0);
     mu_prior_sample = vsamp(hyperp.ms_0', hyperp.Cs_0, priorSampleSize);
     
%      ms_0 = sum(tmpMus,1);
%         hyperp.ms_0 = ms_0/norm(ms_0);
%         mu_prior_sample = vsamp(hyperp.ms_0', hyperp.Cs_0, priorSampleSize);
    end
    k = length(unique(z));
    
    %can show that derivative is guaranteed to be positive / negative at
    %these points
    deriv_up = 2 / (N - k + 3/2);
    deriv_down = k * N/ (N - k + 1);
    
    %this is the version with a conjugate inverse gamma prior on alpha, as
    %in Rasmussen 2000
    hyperp.alpha = ars(@logalphapdf, {k, N}, 1, [deriv_up deriv_down], [deriv_up inf]);

    %this is the version with a totally non-informative prior
    %params(it).alpha = ars(@logalphapdfNI, {k, n}, 1, [deriv_up deriv_down], [deriv_up inf]);
    
    
    if mod(t, 100) ==0
        
        fprintf("Iteration: %d.\n", t); 
    end
end


function [kk,ll] = sample_z(data, m_, mu_s, kappa_s, hyperpar_, ns)
    D=length(data);
    c = find(m_~=0); % gives indices of non-empty clusters
   % r = sum(m_);
    alpha= hyperpar_.alpha;
    k = length(c);
   % n = m(c).*exp(loggausspdf(repmat(z, 1, length(c))', U_mu(:, c)', U_Sigma(:, :, c))');
    mu_s = mu_s(c, :);
    kappa_s = kappa_s(c);
    %mu_t =zeros(k, 2);
    %kappa_t = zeros(1,k);
%     for i_ = 1: length(c)
%         mu_s(i_,:) = pars_(c(i_)).mus;
%         %mu_t(i,:) = pars(c(i)).mut;
%         kappa_s(i_) = pars_(c(i_)).kappas;
%         %kappa_t(i) = pars(c(i)).kappat;
%     end
    
    
    logPost = log(m_(c))' + logVmf(data', mu_s', kappa_s) ; 
    %ns = 1; %monte carlo sample size
    %[mu_s_,  kappa_s_]=samplePosterior(zeros(1,D), 0, hyperpar, ns);
    rdIdx = randperm(priorSampleSize,ns);
    mu_s_ = mu_prior_sample(rdIdx, :);
    kappa_s_ = kappa_prior_sample(rdIdx);
    
    logPredLik = logVmf(data', mu_s_', kappa_s_') ;
    
    logP_0 = log(alpha) -log(ns) + logsumexp(logPredLik);
    logP = [logPost', logP_0];
    pp = exp(logP - max(logP)); % -max(p) for numerical stability
    pp = pp / sum(pp);
    p0 = pp(length(pp));
    u=rand(1);
    if u<p0
        kk = find(m_==0, 1 );
        ll= logP_0;
    else
        u1 = (u-p0);
        ind = find(cumsum(pp(1:(length(pp)-1)))>=u1, 1 );        
        kk = c(ind);
        ll = logPost(ind);
    end

end

function llh =calLoglik()
     n_k_ = m(ind)';
     kappa_ks = U_kappa(ind); 
     nC=length(ind);
     lambda_s = kappa_ks' .* ss_mat+ hyperp.Cs_0*hyperp.ms_0;
     %lambda_t = kappa_ks(:,2) .* ss_mat(:, (D-1):(D)) + hyperp.Ct_0*hyperp.mt_0;
     llh = nC*(logC_d(D, hyperp.Cs_0)) + sum(n_k_.*logC_d(D, kappa_ks') -logC_d(D, sqrt(sum(lambda_s.^2,2))));
     llh =llh+nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
end

function ll_alpha = calDPLogLik()
    n_k_ = m(ind)';
    nC=length(ind);
    ll_alpha= nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
end

end
% 
% function kk = sample_z(data, m, pars, hyperpar, ns)
%     D=length(data);
%     c = find(m~=0); % gives indices of non-empty clusters
%     r = sum(m);
%     alpha= hyperpar.alpha;
%     k = length(c);
%    % n = m(c).*exp(loggausspdf(repmat(z, 1, length(c))', U_mu(:, c)', U_Sigma(:, :, c))');
%     mu_s = zeros(k, D);
%     kappa_s = zeros(1,k);
%     %mu_t =zeros(k, 2);
%     %kappa_t = zeros(1,k);
%     for i = 1: length(c)
%         mu_s(i,:) = pars(c(i)).mus;
%         %mu_t(i,:) = pars(c(i)).mut;
%         kappa_s(i) = pars(c(i)).kappas;
%         %kappa_t(i) = pars(c(i)).kappat;
%     end
%     
%     
%     logPost = log(m(c))' + logVmf(data', mu_s', kappa_s) ; 
%     %ns = 1; %monte carlo sample size
%     %[mu_s_,  kappa_s_]=samplePosterior(zeros(1,D), 0, hyperpar, ns);
%     rdIdx = randi([1 priorSampleSize],1,ns);
%     mu_s_ = mu_prior_sample(rdIdx, :);
%     kappa_s_ = kappa_prior_sample(rdIdx);
%     
%     logPredLik = logVmf(data', mu_s_', kappa_s_') ;
%     
%     logP_0 = log(alpha) -log(ns) + logsumexp(logPredLik);
%     logP = [logPost', logP_0];
%     pp = exp(logP - max(logP)); % -max(p) for numerical stability
%     pp = pp / sum(pp);
%     p0 = pp(length(pp));
%     u=rand(1);
%     if u<p0
%         kk = find(m==0, 1 );
%     else
%         u1 = (u-p0);
%         ind = find(cumsum(pp(1:(length(pp)-1)))>=u1, 1 );        
%         kk = c(ind);
%     end
% 
% end



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
           kappa_s = slicesample(kappaML(ss, n, D), ns, 'logpdf', f_s, 'burnin', 10);
        end
    catch ME
        kappa_s = kappaML(ss, n, D)*0.9;
    end
    % sample kappa_t
    %f_t = @(x) logPosteriorKappaPdf(x, ss((D-1):D), n, hyperpar.mt_0, hyperpar.Ct_0, hyperpar.a_t, hyperpar.b_t);
    %try
    %    kappa_t = slicesample(kappaML(ss((D-1):(D)), n, 2), ns, 'logpdf', f_t,  'burnin', 10);
    %catch ME
        
    %end
    %k= [kappa_s, kappa_t];
    % sample mu_s
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
        
        mu_s =[];
    end 
    % sample mu_t
    %if n > 0
    %ss_t = kappa_t.* ss((D-1):D) + hyperpar.Ct_0.* hyperpar.mt_0;
    %kappa = norm(ss_t);
    %mu = ss_t / kappa; 
    %mu_t=vsamp(mu', kappa, 1);
    %else
    %    mu_t = vsamp(hyperpar.mt_0', hyperpar.Ct_0, ns);
    %end
    %m = [mu_s, mu_t];
end







function x = drawGamma(shape, mean)
% Draw a gamma-distributed random variable having shape and mean
% parameters given by the arguments.  Translate's Rasmussen's shape
% and mean notation to mathworld's and mathworks' alpha and theta
% notation.  When rasmussen writes G(beta, w^-1), matlab expects
% G(beta, w^-1/beta).
x = gamrnd(shape/2, 2*mean./shape);

end


function x = drawGammaWithRate(shape, rate)
% Draw a gamma-distributed random variable having shape and rate; i.e. inverse
% scale
x = gamrnd(shape, 1./rate);

end
