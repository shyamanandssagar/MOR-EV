function result = PolyRidge_solve(data)

    Phi_tr = poly_expand(data.Xtr_n, 3);
    Phi_te = poly_expand(data.Xte_n, 3);
    mphi = mean(Phi_tr);
    sphi = std(Phi_tr) + 1e-10;
    Phi_tr_n = (Phi_tr - mphi) ./ sphi;
    Phi_te_n = (Phi_te - mphi) ./ sphi;

    alpha = 10;
    w = (Phi_tr_n' * Phi_tr_n + alpha * eye(size(Phi_tr_n, 2))) \ (Phi_tr_n' * data.ytr);

    yp = Phi_te_n * w;
    rmse = sqrt(mean((data.yte - yp).^2));
    r2   = 1 - sum((data.yte - yp).^2) / sum((data.yte - mean(data.yte)).^2);
    fprintf('RMSE = %.4f   R2 = %.4f\n', rmse, r2);

    fpred = @(p) ((poly_expand((p - data.mu) ./ data.sg, 3) - mphi) ./ sphi) * w;
    best = inf;  bp = [];
    for k = 1:data.nstarts
        x0 = data.lb + (data.ub - data.lb) .* rand(1, 3);
        [p, v] = fmincon(fpred, x0, [], [], [], [], ...
            data.lb, data.ub, [], data.opts);
        if v < best && all(p > 0)
            best = v;  bp = p;
        end
    end

    result = eval_tf(bp, data.sysH, data.t, data.yH, data.DC, 'PolyRidge');
    result.RMSE = rmse;  result.R2 = r2;
end
