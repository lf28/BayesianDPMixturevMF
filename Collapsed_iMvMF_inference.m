function [z_sample, logllh, kappa_sample] = Collapsed_iMvMF_inference(Y, niter, calLogllh, keepKappa)
% The algorithm implements a Gibbs sampler for infinite mixture of
% von Mises Fiser distribution. The state of the chain include z and kappa; the mean directional
% parameters mu are integrated out;
% Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk;
% distributable under GPL, see README.txt
% Please cite the following paper if needed (also for more details)
% L. Fang, J. Ye and S. Dobson, "Sensor-Based Human Activity Mining Using Dirichlet 
% Process Mixtures of Directional Statistical Models," 2019 IEEE International Conference on 
% Data Science and Advanced Analytics (DSAA), Washington, DC, USA, 2019, pp. 154-163.
% the state of the chain include z and phi
% Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk;
% distributable under GPL, see README.txt

[N, D]=size(Y);
ss_y = sum(Y, 1);
hyperp.alpha = 1;
hyperp.ms_0=ss_y/norm(ss_y); 
hyperp.Cs_0=1; 
%hyperp.mt_0=ss_y((D-1):(D))/norm(ss_y((D-1):(D))); 
%hyperp.Ct_0= 0.5;
%rbar_s = norm(ss_y(1:(D-2)))/N;
%rbar_t = norm(ss_y((D-1):(D)))/N;
%kappaML_s = rbar_s * ((D-2)-rbar_s^2)/(1-rbar_s^2);
%kappaML_t = rbar_t * (2-rbar_t^2)/(1-rbar_t^2);
mc_n= 50;
hyperp.a_s=3;
hyperp.b_s=0.1;
%hyperp.a_t=1;
%hyperp.b_t=1/5;
uss_0.nu = 0;
uss_0.SS = zeros(1,D);
logllh = zeros(1, niter);

priorSampleSize = 1000;
[mu_prior_sample,  kappa_prior_sample]=samplePosterior(uss_0, hyperp, priorSampleSize, true);

z_sample = zeros(N, niter);
kappa_sample = zeros(N, niter); 

% N is the number of data items
% P is the total dimension   
%% Initialise the state of the chain

% initialize the membership z_i for i =1,...,N
initialK = 4;
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


%U_mu = zeros(N,D); %**
U_kappa = zeros(1,N); %**
U_SS = struct('nu', cell(N,1), 'SS', cell(N,1)); %**

for j = 1:K
   idx = find(z==Ks(j));
   m(Ks(j)) = length(idx);
   ss = sum(Y(idx,:), 1); 
   nj = length(idx);
   U_SS(Ks(j)).nu = nj;
   U_SS(Ks(j)).SS = ss;
   [~,  kappa_s]=samplePosterior(U_SS(Ks(j)), hyperp, 1, false);
   %pars(Ks(j)) = struct('mus', mu_s,'kappas', kappa_s);
   %U_mu(Ks(j),:) = mu_s;
   U_kappa(Ks(j))= kappa_s; 
end
if keepKappa
kappa_sample(:,1) = U_kappa';
end
%optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
if calLogllh
ind = find(m);
ss_mat=cell2mat({U_SS(ind).SS}');
logllh(1) =calLoglik();
end

%% Gibbs steps start 



for t = 2: niter
    
    %update class membership
    for i = 1:N
         m(z(i)) = m(z(i)) - 1;
         U_SS(z(i)) = downdate_SS(Y(i,:), U_SS(z(i)));
         z(i) = sample_z(Y(i,:), m, U_kappa, hyperp, mc_n, U_SS);
         m(z(i)) = m(z(i)) + 1;
         
         if m(z(i))>1
            U_SS(z(i)) = update_SS(Y(i,:), U_SS(z(i)));
         else
            U_SS(z(i)) = update_SS(Y(i,:), uss_0);
            [~,  U_kappa(z(i))] = samplePosterior(U_SS(z(i)), hyperp, 1, false);
           %[U_mu(:, z(i)), U_Sigma(:, :, c(k))] = normalinvwishrnd(U_SS(z(i)));
          end       
    end
    
    z_sample(:, t) = z;

    %update component parameters

    ind = find(m);
    for j=1:length(ind)
        %idx = find(z==ind(j));
        %ss = sum(Y(idx,:), 1);  
        %n_ = length(idx);
        %if n_ >1 % can skip this update step if n_ =1 ;
            %[tmpMus(j,:),  tmpkappa_s]=samplePosterior(ss, n_, hyperp, 1);
            [~,  U_kappa(ind(j))]=samplePosterior(U_SS(ind(j)), hyperp, 1, false);
            %pars(ind(j)) = struct('mus', tmpMus(j,:),  'kappas', tmpkappa_s);
        %end
    end
    if keepKappa
        kappa_sample(:,t) = U_kappa';
    end
    ss_mat=cell2mat({U_SS(ind).SS}');
    
    %
    if calLogllh
        logllh(t) =calLoglik(); 
    end

    %optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
    

    updateHyperPrior = false;
    
    if updateHyperPrior 
     ms_0 = sum(normr(ss_mat), 1);
     ms_0 = ms_0/norm(ms_0);
%    ms_0 = sum(tmpMus,1);
     hyperp.ms_0 = ms_0/norm(ms_0);
     mu_prior_sample = vsamp(hyperp.ms_0', hyperp.Cs_0, priorSampleSize);
    
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
     
    end
    if mod(t, 100) ==0
        %ms_0 = sum(tmpMus,1);
        %hyperp.ms_0 = ms_0/norm(ms_0);
        %mu_prior_sample = vsamp(hyperp.ms_0', hyperp.Cs_0, priorSampleSize);
        fprintf("Iteration: %d.\n", t); 
    end
end


function kk = sample_z(data, m_, kappas_, hyperpar_, ns, USS)
    D=length(data);
    c = find(m_~=0); % gives indices of non-empty clusters
    %r = sum(m_);
    alpha= hyperpar_.alpha;
    %k = length(c);
   % n = m(c).*exp(loggausspdf(repmat(z, 1, length(c))', U_mu(:, c)', U_Sigma(:, :, c))');
    %mu_s = zeros(k, D);
    %kappa_s = zeros(1,k);
    %mu_t =zeros(k, 2);
    %kappa_t = zeros(1,k);
%     for i_ = 1: length(c)
%         mu_s(i_,:) = pars_(c(i_)).mus;
%         %mu_t(i,:) = pars(c(i)).mut;
%         kappa_s(i_) = pars_(c(i_)).kappas;
%         %kappa_t(i) = pars(c(i)).kappat;
%     end
    USS = USS(c);
    kappas_ = kappas_(c); % find non empty clusters kappas
    ss_s = kappas_' .* cell2mat({USS.SS}') + hyperpar_.Cs_0*hyperpar_.ms_0;
    ss_s_with_data = ss_s + kappas_'.* data;
    logPost = log(m_(c))'+ logC_d(D, kappas_)' + logC_d(D, sqrt(sum(ss_s.^2,2))) - logC_d(D, sqrt(sum(ss_s_with_data.^2,2)));

    %logPost = log(m_(c))' + logVmf(data', mu_s', kappa_s) ; 
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
    else
        u1 = (u-p0);
        ind = find(cumsum(pp(1:(length(pp)-1)))>=u1, 1 );        
        kk = c(ind);
    end

end

function llh =calLoglik_2()
     n_k_ = m(ind)';
     kappa_ks = U_kappa(ind); 
     nC=length(ind);
     lambda_s = kappa_ks' .* ss_mat+ hyperp.Cs_0*hyperp.ms_0;
     %lambda_t = kappa_ks(:,2) .* ss_mat(:, (D-1):(D)) + hyperp.Ct_0*hyperp.mt_0;
     llh = nC*(logC_d(D, hyperp.Cs_0)) + sum(n_k_.*logC_d(D, kappa_ks') -logC_d(D, sqrt(sum(lambda_s.^2,2))));
     llh =llh+nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
end


function llh =calLoglik()     
     n_k_ = m(ind)';
     nC=length(ind);
    ns=mc_n;
    hdler = @(x) mcll(x, n_k_);
     rdIdx = randperm(priorSampleSize, ns);
     kappa_spls = kappa_prior_sample(rdIdx)';
     kappa_spls = repelem(kappa_spls, nC, 1);
     kappa_spls = num2cell(kappa_spls,1);
     rst=cellfun(hdler, kappa_spls, 'UniformOutput', false); 
     rst=cell2mat(rst);
     rst=(-1)*log(ns)+logsumexp(rst')';
     llh=nC*(logC_d(D, hyperp.Cs_0)) + sum(rst) ;
     llh =llh+nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
end


function rst=mcll(kappa_ks, n_k_)
lambda_s = kappa_ks .* ss_mat+ hyperp.Cs_0*hyperp.ms_0;
rst=n_k_.*logC_d(D, kappa_ks) -logC_d(D, sqrt(sum(lambda_s.^2,2)));
end
end


