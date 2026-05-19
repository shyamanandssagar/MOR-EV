function idx = eval_tf(bp, sysH, t, yH, DC, name)

    a2 = bp(1);
    a1 = bp(2);
    a0 = bp(3);
    b0 = DC * a0;

    sysR = tf(b0, [a2 a1 a0]);
    yR = step(sysR, t);
    e = abs(yH - yR);

    idx.a2 = a2;  idx.a1 = a1;  idx.a0 = a0;  idx.b0 = b0;
    idx.IAE  = trapz(t, e);
    idx.ISE  = trapz(t, e.^2);
    idx.ITAE = trapz(t, t .* e);
    idx.ITSE = trapz(t, t .* e.^2);

    fprintf('G_%s = %.4f / (%.4f s^2 + %.4f s + %.4f)\n', name, b0, a2, a1, a0);
    fprintf('IAE = %.4f   ISE = %.4f   ITAE = %.4f   ITSE = %.4f\n', ...
        idx.IAE, idx.ISE, idx.ITAE, idx.ITSE);
end
