function m = train_xgb(X, y, T, lr, depth, lambda)

    m.F0 = mean(y);
    m.lr = lr;
    m.trees = cell(1, T);

    F = m.F0 * ones(size(y));
    for t = 1:T
        g = F - y;
        h = ones(size(y));
        target = -g ./ (h + lambda);
        tree = fitrtree(X, target, 'MaxDepth', depth);
        m.trees{t} = tree;
        F = F + lr * predict(tree, X);
    end
end
