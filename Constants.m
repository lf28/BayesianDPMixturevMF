classdef Constants
    properties (Constant = true)
        KAPPA_MAX = 300;
        KAPPA_MAX_VONMISES = 500;
        INIT_KAPPA = 8;
        
        
        
%        KAPPA_MAX = 500;
%       KAPPA_MAX_VONMISES = 500;
%        KAPPA_MIN_UNIFORM = 0.01;
        KAPPA_MIN_UNIFORM = 0.02;

 %       KAPPA_MIN_UNIFORM_VM = 0.01; %change to 0.1
        KAPPA_MIN_UNIFORM_VM = 0.02;
        EPSILON = realmin; % smallest positive normalized floating-point number double
        MAX_ITER = 20;
        MAX_ITEREM = 100;
        GAUSSIAN_REGU = 10e-8;
        %VMF_REGU = 10e-4;
        %SIGN_UNKNOWN_S1 = 1;
        SIGN_UNKNOWN_S1 = -1;
        %SIGN_UNKNOWN_S2 = 1;
        SIGN_UNKNOWN_S2 = -1;
        BSL_MULTIPLIER = 2;
        BSL_CHISQAURED = 0.975;
    end
end