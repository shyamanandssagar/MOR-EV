function m = train_lgbm(X, y, T, lr, nleaves)

    m.F0 = mean(y);
    m.lr = lr;
    m.trees = cell(1, T);

    F = m.F0 * ones(size(y));
    for t = 1:T
        r = y - F;
        [~, idx] = sort(abs(r), 'descend');
        nL = ceil(0.2 * length(r));
        nS = ceil(0.1 * (length(r) - nL));
        sel = [idx(1:nL); idx(nL + randperm(length(r) - nL, nS))];
        tree = fitrtree(X(sel, :), r(sel), 'MaxNumSplits', nleaves - 1);
        m.trees{t} = tree;
        F = F + lr * predict(tree, X);
    end
end
