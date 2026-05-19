function [w, Sw, lambda] = bayes_ridge(X, y, maxIt, tol)

    [n, p] = size(X);
    alpha  = 1e-6;
    lambda = 1e-6;
    mw = zeros(p, 1);
    Sw = eye(p);

    for it = 1:maxIt
        a_old = alpha;
        Sw = inv(lambda * (X' * X) + alpha * eye(p));
        mw = lambda * Sw * X' * y;
        gamma  = p - alpha * trace(Sw);
        alpha  = gamma / (mw' * mw);
        lambda = (n - gamma) / sum((y - X * mw).^2);
        if abs(alpha - a_old) < tol
            break;
        end
    end

    w = mw;
end
