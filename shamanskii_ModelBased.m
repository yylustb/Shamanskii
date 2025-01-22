function shamanskii_ModelBased(m,P0)
% m: inner iteration 
% P0: initial iterative matrix
%% initialization
    if nargin < 1
        m = 3; 
    end
    % LQR setting
    A = [-1.01887 0.90506 -0.00215;0.82225 -1.07741 -0.17555;0 0 -1];
    B = [0;0;1];
    Q = eye(3);
    R = 1;    
    % solve algebraic Riccati equation 
    [K_opt,P_opt] = lqr(A,B,Q,R);
    K_opt=-K_opt;        
    % initial iterative matrix
    if nargin < 2
        K0 = [13.1166 13.8704 -2.9037];
        P0 = lyap( (A+B*K0)', Q+K0'*R*K0 );
    end    
    % save iterative process
    piter_save = norm(P0 - P_opt);    
%% main loop 
    P_i = P0;
    kP_0 = K_operator(P0 ,B,R);
    kP_i = K_operator(P_i,B,R);
    for i = 1:11
        P_i0 = P_i;      
        P_ij = P_i;      
        for j = 0:m-1    
            delta = lyap( A_operator(P_i0,A,B,R)', Ric_operator(P_ij,A,B,Q,R));
            P_ij = P_ij + delta;     
            kP_ij = K_operator(P_ij,B,R);
        end

        P_i = P_ij;                  
        kP_i = K_operator(P_i,B,R);       
        
        piter_save = [piter_save;norm(P_i - P_opt)];
    end

    figure
    hold on
    plot(0:10,piter_save(1:11),'-o','Linewidth',2)
    xlabel( 'Iteration Index' , 'Interpreter' , 'latex' , 'FontSize' , 12 ) ; 
    ylabel( '$\left\| {{P_i} - {P^*}} \right\|$', 'Interpreter' , 'latex'  , 'FontSize' , 12 ) ; 
    title('Shamanskii Iteration with m=2', 'Interpreter' , 'latex' , 'FontSize' , 12 )
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
