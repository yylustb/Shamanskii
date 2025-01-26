function [piter_save,riter_save,iter] = shamanskii_ModelBased(m,P0)
% m: inner iteration 
% P0: initial iterative matrix
%% initialization
    if nargin < 1
        m = 3; 
    end
    % LQR setting
    [A,B,Q,R] = lqr_model;
    % solve algebraic Riccati equation 
    [K_opt,P_opt] = lqr(A,B,Q,R);
    K_opt=-K_opt;   
    % initial iterative matrix
    if nargin < 2
        %K0 = [0.2020 0.6051 -11.3408 -1.8520];
        K0 = [827.5591 258.7122 -619.4836 -116.6448];
        P0 = lyap( (A+B*K0)', Q+K0'*R*K0 );
    end
    % save iterative process
    piter_save = norm(P0 - P_opt);
    riter_save = norm(Ric_operator(P0,A,B,Q,R));
%% main loop 
    P_i = P0;
    iter = 0;
    while piter_save(end)>1e-10
        P_i0 = P_i;      
        P_ij = P_i;      
        for j = 0:m-1    
            delta = lyap( A_operator(P_i0,A,B,R)', Ric_operator(P_ij,A,B,Q,R));
            P_ij = P_ij + delta;     
        end

        P_i = P_ij;
        piter_save = [piter_save;norm(P_i - P_opt)];
        riter_save = [riter_save;norm(Ric_operator(P_i,A,B,Q,R))];
        iter = iter + 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,B,Q,R] = lqr_model
    M = .5;
    m = 0.2;
    b = 0.1;
    I = 0.006;
    g = 9.8;
    l = 0.3;
    p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices    
    A = [0      1              0           0;
         0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
         0      0              0           1;
         0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
    B = [     0;
         (I+m*l^2)/p;
              0;
            m*l/p];
    Q=diag([1,1,1,1]);
    R=1;
end

function out = A_operator(P,A,B,R)
    out = A+B*K_operator(P,B,R);
end
function out = Dk_operator(K1,K2,R)
    out=(K1-K2)'*R*(K1-K2);
end
function K = K_operator(P,B,R)
    K = -R\B'*P;
end
function out = Ric_operator(P,A,B,Q,R)
    out = A'*P+P*A-P*B/R*B'*P+Q;
end
