def optimize(theta_1, learning_rate, T, hyperparameters, estimator_choice):
    lambda_val, beta1, beta2, epsilon, rho = hyperparameters
    m_t_minus_1 = v_t_minus_1 = h_1_minus_k = 0
    theta_t = theta_1

    for t in range(1, T + 1):
        L_t = compute_minibatch_loss(theta_t)
        g_t = compute_gradients(L_t, theta_t)
        m_t = beta1 * m_t_minus_1 + (1 - beta1) * g_t

        if t % k == 1:
            h_t_hat = estimator(theta_t)
            h_t = beta2 * h_1_minus_k + (1 - beta2) * h_t_hat
        else:
            h_t = h_t_minus_1

        theta_t = theta_t - learning_rate * lambda_val * theta_t
        theta_t_plus_1 = theta_t - learning_rate * clip(m_t / max(h_t, epsilon), rho)
        m_t_minus_1 = m_t
        h_t_minus_1 = h_t

    return theta_t_plus_1
