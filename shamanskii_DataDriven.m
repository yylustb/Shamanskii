function shamanskii_DataDriven(M,K0)
%% initialization
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
    end
    % off-policy RL setting
    Kb = [-0.5412 2.8662 0.0963];    % parameter in behavior policy
    a=(rand(1,100)-.5)*1;            % parameter in behavior policy
    Kn=100;                          % parameter in behavior policy
    % initial condition for ode
    x0=[-2;1;2];
    [xn,un]=size(B);
    xx0=zeros(xn*xn,1);     % initial condition
    xu0=zeros(xn*un,1);     % initial condition
    V0 = 0 ;
    X0=[x0;xx0;xu0;V0];     % initial condition

    Nh=100; % number of sampling period
    h=0.05; % duration of sampling period
    if nargin < 1
        M = 3; % inner loop
    end
    % data memory
    x_save=[];
    t_save=[];
    theta_xx_save=[]; 
    theta_xu_save=[]; 
    theta_x_save=[];  

%% Step 1: Data Collection Phase
    for hter=1:Nh
        t0=(hter-1)*h;
        tf=hter*h;
        tspan=[t0 tf];
        [t,X] = ode45(@(t,X) myode(t,X,A,B,Q,R,Kb,a,Kn),tspan,X0);
        theta_x=alpha_fcn(X(end,1:xn))'-alpha_fcn(X(1,1:xn))';
        theta_xx=X(end,xn+1:xn+xn*xn)-X(1,xn+1:xn+xn*xn);
        theta_xu=X(end,xn+xn*xn+1:xn+xn*xn+xn*un)-X(1,xn+xn*xn+1:xn+xn*xn+xn*un);   
        X0=X(end,:)';
        x_save=[x_save;X(:,1:xn)];
        t_save=[t_save;t];
        theta_xx_save=[theta_xx_save;theta_xx];
        theta_xu_save=[theta_xu_save;theta_xu];
        theta_x_save=[theta_x_save;theta_x];
    end

%% Step 2: Data-driven learning Phase
    xi_dim = (1+xn)*xn/2+xn*un;
    L=size(theta_x_save,1);
    AA0=zeros(L,xi_dim );
    for i=1:L
        A1=theta_x_save(i,:); 
        Ixx=theta_xx_save(i,:);  
        Ixu=theta_xu_save(i,:);  
        A2=-2*Ixx*kron(eye(xn),K0'*R)+2*Ixu*kron(eye(xn),R);  
        AA0(i,:) = [A1 A2];
    end
    
    L=size(theta_xx_save,1);
    b0=zeros(L,1);
    Q_iter=Q+K0'*R*K0;    
    for i=1:L
        Ixx=theta_xx_save(i,:);    
        b=-Ixx*Q_iter(:);        
        b0(i,:) = b;
    end
    XX0=AA0\b0;
    P0_vec=XX0(1:xn*(xn+1)/2);
    kP0_vec=XX0(xn*(xn+1)/2+1:xn*(xn+1)/2+xn*un);    
    P0 = beta_inv_fcn(P0_vec);
    kP0 = reshape(kP0_vec,un,xn);

    k_save = norm(kP0-K_opt);
    p_save = norm(P0-P_opt);

    P_i = P0;
    kP_i = kP0;
    for i=1:11
        P_i0 = P_i;
        P_ij = P_i;
        kP_ij = kP_i;
        xi_dim = (1+xn)*xn/2+xn*un;
        L=size(theta_x_save,1);
        AA=zeros(L,xi_dim );
        for i=1:L
            A1=theta_x_save(i,:);
            Ixx=theta_xx_save(i,:);
            Ixu=theta_xu_save(i,:);
            A2=-2*Ixx*kron(eye(xn),kP_i'*R)+2*Ixu*kron(eye(xn),R);
            AA(i,:) = [A1 A2];
        end
        for j=1:M
            if j == 1
                L=size(theta_xx_save,1);
                b=zeros(L,1);
                Q_iter=Q+kP_i'*R*kP_i;                
                for ii=1:L
                    Ixx=theta_xx_save(ii,:);    
                    c=-Ixx*Q_iter(:);        
                    b(ii,:) = c;
                end
                XX=AA\b;
                P_vec=XX(1:xn*(xn+1)/2);
                kP_vec=XX(xn*(xn+1)/2+1:xn*(xn+1)/2+xn*un);    
                P_ij = beta_inv_fcn(P_vec);
                kP_ij = reshape(kP_vec,un,xn);
            else
                L=size(theta_xx_save,1);
                b=zeros(L,1);
                Q_iter=Q+kP_i'*R*kP_i;                
                for ii=1:L
                    Ixx=theta_xx_save(ii,:);    
                    c=-Ixx*Q_iter(:);        
                    b(ii,:) = c;
                end
                L=size(theta_xx_save,1);
                bb=zeros(L,1);
                D=Dk_operator(kP_i,kP_ij,R);                
                for ii=1:L
                    Ixx=theta_xx_save(ii,:);
                    c=Ixx*D(:);
                    bb(ii,:) = c;
                end
                biter = b + bb;
                XX=AA\biter;
                P_vec=XX(1:xn*(xn+1)/2);
                kP_vec=XX(xn*(xn+1)/2+1:xn*(xn+1)/2+xn*un);    
                P_ij = beta_inv_fcn(P_vec);
                kP_ij = reshape(kP_vec,un,xn);
            end
            
        end
        P_i = P_ij;
        kP_i = kP_ij;     
        k_save = [k_save;norm(kP_i-K_opt)];
        p_save = [p_save;norm(P_i-P_opt)];
    end
    %% Step 3: Policy Implementation Phase
    t0=t_save(end);
    tf=50;
    tspan=[t0 tf];
    X0=X(end,:)';
    a=0;
    Kn=0;
    [t,X] = ode45(@(t,X) myode(t,X,A,B,Q,R,kP_i,a,Kn),tspan,X0);       
    t_save=[t_save;t];
    x_save=[x_save;X(:,1:xn)];


    %% step 4: plot results
    figure
    hold on
    plot(0:10,p_save(1:11),'-o','Linewidth',2)
    xlabel( 'Iteration Index' , 'Interpreter' , 'latex' , 'FontSize' , 12 ) ; 
    ylabel( '$\left\| {{P_i} - {P^*}} \right\|$', 'Interpreter' , 'latex'  , 'FontSize' , 12 ) ; 
    title('Shamanskii Iteration with m=2', 'Interpreter' , 'latex' , 'FontSize' , 12 )
end

function P = beta_inv_fcn(P_vec)
    p_dim=length(P_vec);
    P=[    P_vec(1) .5*P_vec(2) .5*P_vec(3);
        .5*P_vec(2) 1.*P_vec(4) .5*P_vec(5);
        .5*P_vec(3) .5*P_vec(5) 1.*P_vec(6)];
end
function [out] = alpha_fcn(x)
    x_dim = length(x);
    x1=x(1);
    x2=x(2);
    x3=x(3);
    out=[x1^2;x1*x2;x1*x3;x2^2;x2*x3;x3^2];
end
function dX = myode(t,X,A,B,Q,R,K,a,Kn)
    [xn,~]=size(B);
    x=X(1:xn);
    
    u = K*x;
    probe_noise = generate_noise(t,a,Kn);
    u = u+probe_noise;
    r=x'*Q*x+u'*R*u;
    
    dx=A*x+B*u;
    dxx=kron(x',x')';
    dxu=kron(x',u')';
    dV=r;
    dX=[dx;dxx;dxu;dV];
end
function n = generate_noise(t,a,Kn)
    n=0;
    La=length(a);
    for i=a
        n=n+sin(i*t)/La;
    end
    n=Kn*n;
end
function out = Dk_operator(K1,K2,R)
    out=(K1-K2)'*R*(K1-K2);
end