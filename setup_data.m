function data = setup_data()

    data.numH = 0.4158;
    data.denH = [1 3.5 5.5 4.6 2.0 1.1];
    data.sysH = tf(data.numH, data.denH);
    data.DC   = data.numH / data.denH(end);
    data.t    = 0:0.033:100;
    data.yH   = step(data.sysH, data.t);

    fprintf('Reference 5th-order HOEVS\n');
    fprintf('G(s) = %.4f / (s^5 + 3.5s^4 + 5.5s^3 + 4.6s^2 + 2.0s + 1.1)\n', data.numH);
    fprintf('DC gain = %.4f\n\n', data.DC);

    N = 600;
    X = zeros(N, 3);
    y = zeros(N, 1);
    i = 1;
    while i <= N
        a2 = 0.5  + 24.5 * rand;
        a1 = 0.05 + 7.95 * rand;
        a0 = 0.05 + 2.95 * rand;
        b0 = data.DC * a0;
        try
            sysR = tf(b0, [a2 a1 a0]);
            yR = step(sysR, data.t);
            if all(isfinite(yR)) && max(abs(yR)) < 1e4
                X(i, :) = [a2 a1 a0];
                y(i)    = trapz(data.t, abs(data.yH - yR));
                i = i + 1;
            end
        catch
        end
    end

    cv = cvpartition(N, 'HoldOut', 0.2);
    data.Xtr = X(training(cv), :);  data.ytr = y(training(cv));
    data.Xte = X(test(cv), :);      data.yte = y(test(cv));

    data.mu = mean(data.Xtr);
    data.sg = std(data.Xtr);
    data.Xtr_n = (data.Xtr - data.mu) ./ data.sg;
    data.Xte_n = (data.Xte - data.mu) ./ data.sg;

    fprintf('Dataset generated: 480 train / 120 test samples\n');
    fprintf('IAE range: [%.2f, %.2f]\n\n', min(y), max(y));
end
