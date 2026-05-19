function yp = pred_boost(m, X)

    yp = m.F0 * ones(size(X, 1), 1);
    for t = 1:length(m.trees)
        yp = yp + m.lr * predict(m.trees{t}, X);
    end
end
