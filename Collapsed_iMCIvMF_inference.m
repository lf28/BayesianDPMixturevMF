function [z_sample, logllh, kappa_sample] = Collapsed_iMCIvMF_inference(Y, niter, calLogllh, keepKappa)
% The algorithm implements a Gibbs sampler for infinite mixture of conditionally independent
% von Mises Fiser distribution. The state of the chain include z and kappa; the mean directional
% parameters mu are integrated out;

% Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk;
% distributable under GPL, see README.txt
% Please cite the following paper if needed (also for more details)
% L. Fang, J. Ye and S. Dobson, "Sensor-Based Human Activity Mining Using Dirichlet 
% Process Mixtures of Directional Statistical Models," 2019 IEEE International Conference on 
% Data Science and Advanced Analytics (DSAA), Washington, DC, USA, 2019, pp. 154-163.
[N, D]=size(Y);
ss_y = sum(Y, 1);
hyperp.alpha = 1;
hyperp.ms_0=ss_y(1:(D-2))/norm(ss_y(1:(D-2))); 
hyperp.Cs_0=1; 
hyperp.mt_0=ss_y((D-1):(D))/norm(ss_y((D-1):(D))); 
hyperp.Ct_0= 1;
%rbar_s = norm(ss_y(1:(D-2)))/N;
%rbar_t = norm(ss_y((D-1):(D)))/N;
%kappaML_s = rbar_s * ((D-2)-rbar_s^2)/(1-rbar_s^2);
%kappaML_t = rbar_t * (2-rbar_t^2)/(1-rbar_t^2);
mc_n=50;

% hyperp.a_s=16;
% hyperp.b_s=4/5;
% hyperp.a_t=16;
% hyperp.b_t=4/5;

hyperp.a_s=3;
hyperp.b_s=0.1;
hyperp.a_t=3;
hyperp.b_t=0.1;

uss_0.nu = 0;
uss_0.SS = zeros(1,D);
logllh = zeros(1, niter);
priorSampleSize = 1000;
[mu_s_prior_sample,  mu_t_prior_sample, kappa_s_prior_sample, kappa_t_prior_sample]=...
    samplePosterior_CI(uss_0, hyperp, priorSampleSize, true);


z_sample = zeros(N, niter);
kappa_sample = zeros(N, 2, niter); 
% N is the number of data items
% P is the total dimension   
%% Initialise the state of the chain

% initialize the membership z_i for i =1,...,N
initialK = 8;
if initialK > N
   initialK = N; 
end
z = randi(initialK, N, 1);
%z = kmeans(Y, initialK);
z_sample(:,1) =z;
m = zeros(1,N);
% sample phi
Ks = unique(z);
K = length(Ks); 

%Ms = zeros(K, D);
%Kappas = zeros(K, 2);

U_kappa = zeros(N,2); %**
U_SS = struct('nu', cell(N,1), 'SS', cell(N,1)); %**
for j = 1:K
   idx = find(z==Ks(j));
   m(Ks(j)) = length(idx);
   ss = sum(Y(idx,:), 1); 
   nj = length(idx);
   U_SS(Ks(j)).nu = nj;
   U_SS(Ks(j)).SS = ss;
   [~, ~,  kappa_s, kappa_t]=samplePosterior_CI(U_SS(Ks(j)), hyperp, 1, false);
   %pars(Ks(j)) = struct('mus', mu_s, 'mut' , mu_t, 'kappas', kappa_s, 'kappat', kappa_t);
   U_kappa(Ks(j),:)= [kappa_s, kappa_t]; 
end
if keepKappa
    kappa_sample(:,:, 1) = U_kappa;
end
if calLogllh
ind = find(m);
ss_mat=cell2mat({U_SS(ind).SS}');
logllh(1) =calLoglik();
end

%optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t


%% Gibbs steps start 

for t = 2: niter
    
    %update class membership
    for i = 1:N
         m(z(i)) = m(z(i)) - 1;
         U_SS(z(i)) = downdate_SS(Y(i,:), U_SS(z(i)));
         %z(i) = sample_z(Y(i,:), m, pars, hyperp, mc_n);
         z(i) = sample_z(Y(i,:), m, U_kappa, hyperp, mc_n, U_SS);

         m(z(i)) = m(z(i)) + 1;
         
         
         if m(z(i))>1
            U_SS(z(i)) = update_SS(Y(i,:), U_SS(z(i)));
         else
            U_SS(z(i)) = update_SS(Y(i,:), uss_0);
            [~, ~, U_kappa(z(i), 1), U_kappa(z(i), 2)] = samplePosterior_CI(U_SS(z(i)), hyperp, 1, false);
           %[U_mu(:, z(i)), U_Sigma(:, :, c(k))] = normalinvwishrnd(U_SS(z(i)));
          end        
    end
    
    z_sample(:, t) = z;

    %update component parameters
    
    ind = find(m);
    for j=1:length(ind)
        %idx = find(z==ind(j));
        %ss = sum(Y(idx,:), 1);
        %[tmpmu_s, tmpmu_t,  tmpkappa_s, tmpkappa_t]=samplePosterior(ss, length(idx), hyperp, 1);
        %pars(ind(j)) = struct('mus', tmpmu_s, 'mut' , tmpmu_t, 'kappas', tmpkappa_s, 'kappat', tmpkappa_t);
        [~, ~, U_kappa(ind(j), 1), U_kappa(ind(j), 2)]=samplePosterior_CI(U_SS(ind(j)), hyperp, 1, false);
    end
    if keepKappa
    kappa_sample(:,:,t) = U_kappa;
    end
    
     ss_mat=cell2mat({U_SS(ind).SS}');
%
    if calLogllh
        logllh(t) =calLoglik(); 
    end
     
     %optional update hyperparameters i.e. ms_0, Cs_0, mt_0, Ct_0, a_s, b_s, a_t, b_t
     updateHyperPrior =true;
     if updateHyperPrior
     ms_0 = sum(normr(ss_mat(:,1:D-2)), 1);
     mt_0= sum(normr(ss_mat(:,(D-1):D)), 1);
%    ms_0 = sum(tmpMus,1);
     hyperp.ms_0 = ms_0/norm(ms_0);
     hyperp.mt_0 = mt_0/norm(mt_0);
     mu_s_prior_sample = vsamp(hyperp.ms_0', hyperp.Cs_0, priorSampleSize);
     mu_t_prior_sample = vsamp(hyperp.mt_0', hyperp.Ct_0, priorSampleSize);
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
    
    
%     if mod(t, 100) ==0
%        fprintf("Iteration: %d.\n", t); 
%     end
end



function kk = sample_z(data, m_, kappas_, hyperpar_, ns, USS)
    D=length(data);
    c = find(m~=0); % gives indices of non-empty clusters
    alpha= hyperpar_.alpha;
    k = length(c);
    USS = USS(c);
    kappas_ = kappas_(c,:);
    %kappas_s = 
    uss_mat = cell2mat({USS.SS}');
    ss_s = kappas_(:,1) .* uss_mat(:, 1:(D-2)) + hyperpar_.Cs_0*hyperpar_.ms_0;
    ss_t = kappas_(:,2) .* uss_mat(:, (D-1):(D)) + hyperpar_.Ct_0*hyperpar_.mt_0;
    ss_s_with_data = ss_s + kappas_(:,1).* data(1:(D-2));
    ss_t_with_data = ss_t + kappas_(:,2).* data((D-1):(D));
    logPost = log(m_(c))'+ logC_d(D-2, kappas_(:,1)) + logC_d(D-2, sqrt(sum(ss_s.^2,2))) - logC_d(D-2, sqrt(sum(ss_s_with_data.^2,2)))...
        + logC_d(2, kappas_(:,2)) + logC_d(2, sqrt(sum(ss_t.^2,2))) - logC_d(2, sqrt(sum(ss_t_with_data.^2,2)));
   
    
    %logPost = log(m(c))' + logVmf(data(1:(D-2))', mu_s', kappa_s) + logVmf(data((D-1):D)', mu_t', kappa_t); 
    %ns = 1; %monte carlo sample size
    %[mu_s_, mu_t_,  kappa_s_, kappa_t_]=samplePosterior(zeros(1,D), 0, hyperpar, ns);
    
    rdIdx_s = randperm(priorSampleSize,ns);
    rdIdx_t = randperm(priorSampleSize,ns);
    mu_s_ = mu_s_prior_sample(rdIdx_s, :);
    kappa_s_ = kappa_s_prior_sample(rdIdx_s);
    mu_t_ = mu_t_prior_sample(rdIdx_t, :);
    kappa_t_ = kappa_t_prior_sample(rdIdx_t);

    logPredLik = logVmf(data(1:(D-2))', mu_s_', kappa_s_') + logVmf(data((D-1):D)', mu_t_', kappa_t_');
    
    logP_0 = log(alpha) -log(ns) + logsumexp(logPredLik);
    logP = [logPost', logP_0];
    pp = exp(logP - max(logP)); % -max(p) for numerical stability
    pp = pp / sum(pp);
    p0 = pp(length(pp));
    u=rand(1);
    if u<p0
        kk = find(m==0, 1 );
    else
        u1 = (u-p0);
        ind_ = find(cumsum(pp(1:(length(pp)-1)))>=u1, 1 );        
        kk = c(ind_);
    end

end


% function llh =calLoglik_2()
%      n_k_ = m(ind)';
%      kappa_ks = U_kappa(ind,:); 
%      nC=length(ind);
%      lambda_s = kappa_ks(:,1) .* ss_mat(:, 1:(D-2)) + hyperp.Cs_0*hyperp.ms_0;
%      lambda_t = kappa_ks(:,2) .* ss_mat(:, (D-1):(D)) + hyperp.Ct_0*hyperp.mt_0;
%      llh = nC*(logC_d(D-2, hyperp.Cs_0)+logC_d(2, hyperp.Ct_0)) + sum(n_k_.*logC_d(D-2, kappa_ks(:,1)) -logC_d(D-2, sqrt(sum(lambda_s.^2,2))) + n_k_.*logC_d(2, kappa_ks(:,2))...
%          -logC_d(2, sqrt(sum(lambda_t.^2,2))));
%      llh =llh+nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
% end


function llh =calLoglik()     
     n_k_ = m(ind)';
     nC=length(ind);
     ns=mc_n;
     hdler = @(x) mcll(x, n_k_);
     rdIdx = randperm(priorSampleSize, ns);
     kappa_spls_s = kappa_s_prior_sample(rdIdx);
     kappa_spls_t = kappa_t_prior_sample(rdIdx);
     kappa_spls = [kappa_spls_s, kappa_spls_t];
     kappa_spls = num2cell(kappa_spls, 2);
     kappa_spls = cellfun(@(x) repelem(x, nC,1), kappa_spls, 'UniformOutput', false);
     rst=cellfun(hdler, kappa_spls', 'UniformOutput', false); 
     rst=cell2mat(rst);
     rst=(-1)*log(ns)+logsumexp(rst')';
     llh=nC*(logC_d(D-2, hyperp.Cs_0)+logC_d(2, hyperp.Ct_0)) + sum(rst) ;
     llh =llh+nC*log(hyperp.alpha) + sum(log(factorial(n_k_-1))) - sum(log((0:(N-1))+hyperp.alpha));
end


function rst=mcll(kappa_ks, n_k_)
lambda_s = kappa_ks(:,1) .* ss_mat(:, 1:(D-2)) + hyperp.Cs_0*hyperp.ms_0;
lambda_t = kappa_ks(:,2) .* ss_mat(:, (D-1):(D)) + hyperp.Ct_0*hyperp.mt_0;
rst=n_k_.*logC_d(D-2, kappa_ks(:,1)) -logC_d(D-2, sqrt(sum(lambda_s.^2,2))) + n_k_.*logC_d(2, kappa_ks(:,2))...
         -logC_d(2, sqrt(sum(lambda_t.^2,2)));
end


end





