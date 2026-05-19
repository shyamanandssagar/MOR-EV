MACHINE LEARNING BASED MODEL ORDER REDUCTION OF AN ELECTRIC VEHICLE SYSTEM



HOW TO RUN
    1. Open MATLAB (R2021a or later).
    2. MAKE A FOLDer and set all files in that folder suppose MOR EV PROJECT Set the current folder to MOR_EV_Project.
    3. Run: main


REQUIRED TOOLBOXES
    Statistics and Machine Learning Toolbox
    Control System Toolbox
    Optimization Toolbox


FILES
    main.m                  Main runner script
    setup_data.m            Defines 5th-order HOEVS, generates 600 stable samples

    SVR_solve.m             Support Vector Regression surrogate
    GPR_solve.m             Gaussian Process Regression surrogate
    RF_solve.m              Random Forest surrogate
    MLP_solve.m             Multi-Layer Perceptron surrogate
    XGBoost_solve.m         XGBoost surrogate
    LightGBM_solve.m        LightGBM surrogate
    KNN_solve.m             K-Nearest Neighbours surrogate
    PolyRidge_solve.m       Polynomial Ridge surrogate
    BayesianRidge_solve.m   Bayesian Ridge surrogate
    GBR_solve.m             Gradient Boosting Regression surrogate

    eval_tf.m               Computes IAE/ISE/ITAE/ITSE for a reduced TF
    poly_expand.m           Polynomial feature expansion up to degree 3
    bayes_ridge.m           EM iteration for Bayesian Ridge
    train_xgb.m             Custom XGBoost training loop
    train_lgbm.m            Custom LightGBM with GOSS sampling
    pred_boost.m            Prediction for boosting ensembles
    tb_pred.m               TreeBagger prediction wrapper


REFERENCE SYSTEM (5th-order HOEVS)
    G(s) = 0.4158 / (s^5 + 3.5 s^4 + 5.5 s^3 + 4.6 s^2 + 2.0 s + 1.1)
    DC gain = 0.3780

REDUCED 2nd-ORDER MODEL FORM
    G_r(s) = b0 / (a2 s^2 + a1 s + a0)
    subject to:
        a2, a1, a0 > 0    (Hurwitz stability)
        b0 = 0.3780 * a0   (DC gain matching)


OUTPUT
    Console: per-method TF, IAE, ISE, ITAE, ITSE and surrogate R2
    Figure: step response of HOEVS overlaid with all 10 reduced models
    File:   mor_results.mat   contains the complete results struct
