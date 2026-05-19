function v = tb_pred(mdl, X)

    yp = predict(mdl, X);
    if iscell(yp)
        yp = str2double(yp);
    end
    v = yp;
end
