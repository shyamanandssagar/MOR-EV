function Xp = poly_expand(X, d)

    Xp = X;
    nf = size(X, 2);

    if d < 2
        return;
    end

    for i = 1:nf
        for j = i:nf
            Xp = [Xp, X(:, i) .* X(:, j)];
            if d >= 3
                for k = j:nf
                    Xp = [Xp, X(:, i) .* X(:, j) .* X(:, k)];
                end
            end
        end
    end
end
