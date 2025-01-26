clear
clc
close all
%% 
[p_chord,r_chord]               = chord_ModelBased;
[p_newton,r_newton]             = shamanskii_ModelBased(1);
[p_chevshev,r_chevshev]         = shamanskii_ModelBased(2);
[p_shamanskii_3,r_shamanskii_3] = shamanskii_ModelBased(3);
[p_shamanskii_4,r_shamanskii_4] = shamanskii_ModelBased(4);
%% Figure 1
semilogy(0:15,p_chord(1:16),'-x', 'LineWidth',2)
hold on
semilogy(0:length(p_newton)-1,p_newton,'-*', 'LineWidth',2)
semilogy(0:length(p_chevshev)-1,p_chevshev,'-s', 'LineWidth',2)
semilogy(0:length(p_shamanskii_3)-1,p_shamanskii_3,'-^', 'LineWidth',2)
semilogy( 0:length(p_shamanskii_4)-1,p_shamanskii_4,'-o', 'LineWidth',2)
set(gcf, 'Position', [0, 0, 1000, 600]); 
set(gca,'FontSize',12);
xlabel( 'Iteration Index' , 'Interpreter' , 'latex' , 'FontSize' , 25 ) ; 
ylabel( '$\left\| P_i - P^* \right\|$' , 'Interpreter' , 'latex' , 'FontSize' , 25 ) ; 
h=legend('Chord Iteration',...
         'Newton Iteration ($p=1$)',...
         'Chebschev Iteration ($p=2$)',...
         'Shamanskii Iteration ($p=3$)',...
         'Shamanskii  Iteration ($p=4$)',...
         'FontSize', 14,  'Location', 'southwest');
set(h,'Interpreter','latex')
%% Figure 2
m=4;
[p_save,r_save,t_save,x_save] = shamanskii_DataDriven(m);



figure
plot(t_save,x_save(:,1),':','LineWidth',2)
hold on
plot(t_save,x_save(:,2),'--','LineWidth',2)
plot(t_save,x_save(:,3),'-.','LineWidth',2)
plot(t_save,x_save(:,4),'-','LineWidth',2)
xlabel( 'Time/sec' , 'Interpreter' , 'latex' , 'FontSize' , 25 ) ; 
h=legend('$x_1(t)$','$x_2(t)$','$x_3(t)$','$x_4(t)$', 'FontSize', 18,  'Location', 'northeast');
set(h,'Interpreter','latex')
set(gcf, 'Position', [100, 100, 1000, 600]); 
%% Figure 3
n=4;
L=6;
y = zeros(L,1);
for q=1:L
    y(q) = (q+1)^(1/((29+13*q)*n^3)) ;
end

x = 0:L-1;
figure
bar( x(1:6),y(1:6));
set(gca, 'YScale', 'log');
set(gca,'FontSize',20);
xlabel('Value of $p$ in Shamanskii Iteration','interpreter','latex', 'FontSize', 25); 
title('Computational Efficiency Index ${ord^{\frac{1}{{op}}}} = {\left( {p + 1} \right)^{\frac{1}{{\left( {29 + 13p} \right){n^3}}}}}$','interpreter','latex', 'FontSize', 25); % 图标题
ylim([1.0002 1.00033])
set(gcf, 'Position', [100, 100, 1000, 600]); 