clc; clear; close all;
rng(42);

data = setup_data();
data.lb = [0.5 0.1 0.1];
data.ub = [20 6 2.5];
data.opts = optimoptions('fmincon','Display','off','MaxIterations',1500);
data.nstarts = 30;

results = struct();

fprintf('\n----- SVR -----\n');
results.SVR = SVR_solve(data);

fprintf('\n----- GPR -----\n');
results.GPR = GPR_solve(data);

fprintf('\n----- Random Forest -----\n');
results.RF = RF_solve(data);

fprintf('\n----- MLP -----\n');
results.MLP = MLP_solve(data);

fprintf('\n----- XGBoost -----\n');
results.XGB = XGBoost_solve(data);

fprintf('\n----- LightGBM -----\n');
results.LGBM = LightGBM_solve(data);

fprintf('\n----- KNN -----\n');
results.KNN = KNN_solve(data);

fprintf('\n----- Polynomial Ridge -----\n');
results.PolyRidge = PolyRidge_solve(data);

fprintf('\n----- Bayesian Ridge -----\n');
results.BayesRidge = BayesianRidge_solve(data);

fprintf('\n----- GBR -----\n');
results.GBR = GBR_solve(data);


fprintf('\n----- Summary -----\n');
methods = fieldnames(results);
fprintf('%-14s %-10s %-10s %-10s %-10s %-10s\n','Method','IAE','ISE','ITAE','ITSE','R2');
for i = 1:length(methods)
    r = results.(methods{i});
    fprintf('%-14s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        methods{i}, r.IAE, r.ISE, r.ITAE, r.ITSE, r.R2);
end


figure('Position',[100 100 1100 700]);
plot(data.t, data.yH, 'b-', 'LineWidth', 2.0); hold on;
cols = lines(length(methods));
legendLabels = cell(length(methods)+1, 1);
legendLabels{1} = 'HOEVS (5th order)';
for i = 1:length(methods)
    r = results.(methods{i});
    sysR = tf(r.b0, [r.a2 r.a1 r.a0]);
    yR = step(sysR, data.t);
    plot(data.t, yR, '--', 'Color', cols(i,:), 'LineWidth', 1.2);
    legendLabels{i+1} = methods{i};
end
legend(legendLabels, 'Location','best');
xlabel('Time (s)');
ylabel('Amplitude');
title('Step Response Comparison: HOEVS vs Reduced 2nd-order Models');
grid on;

save('mor_results.mat','results','data');
