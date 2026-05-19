function result = RF_solve(data)

    mdl = TreeBagger(150, data.Xtr, data.ytr, 'Method','regression', ...
        'MinLeafSize', 5);

    yp = predict(mdl, data.Xte);
    if iscell(yp), yp = str2double(yp); end
    rmse = sqrt(mean((data.yte - yp).^2));
    r2   = 1 - sum((data.yte - yp).^2) / sum((data.yte - mean(data.yte)).^2);
    fprintf('RMSE = %.4f   R2 = %.4f\n', rmse, r2);

    best = inf;  bp = [];
    for k = 1:data.nstarts
        x0 = data.lb + (data.ub - data.lb) .* rand(1, 3);
        [p, v] = fmincon(@(p) tb_pred(mdl, p), ...
            x0, [], [], [], [], data.lb, data.ub, [], data.opts);
        if v < best && all(p > 0)
            best = v;  bp = p;
        end
    end

    result = eval_tf(bp, data.sysH, data.t, data.yH, data.DC, 'RF');
    result.RMSE = rmse;  result.R2 = r2;
end
