function [X_tensor] = LMELTV(Y_tensor, gamma_M, lambda_M, r, w)
% LMELTV
% Hyperspectral image denoising via enhanced Laplacian total variation regularizer
%
% Input:
%   Y_tensor  - noisy HSI tensor \mathcal{Y} \in R^{m x n x B}
%   gamma_M   - ELTV regularization parameter in the paper
%   lambda_M  - sparse noise regularization parameter in the paper
%   r         - rank bound of X
%   w         - weights [w_h, w_v, w_t]
%
% Output:
%   X_tensor  - denoised HSI tensor \mathcal{X}

%% Preprocessing
[m, n, B] = size(Y_tensor);
Y = reshape(Y_tensor, [m*n, B]);
normY = norm(Y, 'fro');

%% Parameters
sv = 10;

tmp = 25 / (normY^2);
mu_Q = tmp;          % penalty parameter for constraint Z = X
mu_P = tmp;          % penalty parameter for constraint Y = X + S
mu_T = tmp;          % penalty parameter for constraint D_w(Z) = G

mu_max  = 1e6;
rho     = 1.4;
tol_Q   = 2e-6;
tol_P   = 2e-6;
maxIter = 100;

%% Initialization
X = zeros(size(Y));          % low-rank component X
Z = zeros(size(Y));          % auxiliary variable Z
S = sparse(Z);               % sparse noise S

Q = zeros(size(Y));          % multiplier for Z - X
P = zeros(size(Y));          % multiplier for Y - X - S

G_h = zeros(m, n, B);        % horizontal gradient auxiliary variable
G_v = zeros(m, n, B);        % vertical gradient auxiliary variable
G_t = zeros(m, n, B);        % spectral gradient auxiliary variable

T_h = zeros(m, n, B);        % multiplier for D_h(Z) - G_h
T_v = zeros(m, n, B);        % multiplier for D_v(Z) - G_v
T_t = zeros(m, n, B);        % multiplier for D_t(Z) - G_t

%% Weighted difference operators D_w and D_w^*
[D_w, D_w_adj] = defDDt(w);

eigDhDv = abs(w(1) * fftn([1 -1],  [m n B])).^2 + ...
          abs(w(2) * fftn([1 -1]', [m n B])).^2;

if B > 1
    d_tmp = zeros(1,1,B);
    d_tmp(1,1,1) = 1;
    d_tmp(1,1,2) = -1;
    eigDt = abs(w(3) * fftn(d_tmp, [m n B])).^2;
else
    eigDt = 0;
end

%% Main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;

    %% (1) Update X
    X = (mu_Q * Z + mu_P * (Y - S) + Q + P) / (mu_Q + mu_P);
    [X, sv] = prox_nuclear(X, mu_Q, mu_P, B, sv, r);

    %% (2) Update Z
    J1 = reshape(X - Q / mu_Q, m, n, B);
    J2 = D_w_adj( ...
        G_h - T_h / mu_T, ...
        G_v - T_v / mu_T, ...
        G_t - T_t / mu_T);

    rhs = fftn((mu_Q / mu_T) * J1 + J2);
    lhs = (mu_Q / mu_T) + eigDhDv + eigDt;
    Z_tensor = real(ifftn(rhs ./ lhs));
    Z = reshape(Z_tensor, [m*n, B]);

    %% (3) Update G = [G_h, G_v, G_t]
    [D_h_Z, D_v_Z, D_t_Z] = D_w(Z_tensor);

    G_h = prox_L23(D_h_Z + T_h / mu_T, gamma_M / mu_T);
    G_v = prox_L23(D_v_Z + T_v / mu_T, gamma_M / mu_T);

    if w(3) == 0
        G_t = 0;
    else
        G_t = prox_L23(D_t_Z + T_t / mu_T, gamma_M / mu_T);
    end

    %% (4) Update S
    S = softthre(Y - X + P / mu_P, lambda_M / mu_P);

    %% Stop criterion
    leq_Q = Z - X;          % Z - X
    leq_P = Y - X - S;      % Y - X - S

    stopC_Q = max(max(abs(leq_Q)));
    stopC_P = norm(leq_P, 'fro') / normY;

    %% (5) Update multipliers P, Q, T
    Q = Q + mu_Q * leq_Q;
    P = P + mu_P * leq_P;

    T_h = T_h + mu_T * (D_h_Z - G_h);
    T_v = T_v + mu_T * (D_v_Z - G_v);
    T_t = T_t + mu_T * (D_t_Z - G_t);

    %% (6) Update penalty parameters
    mu_Q = min(mu_max, mu_Q * rho);
    mu_P = min(mu_max, mu_P * rho);
    mu_T = min(mu_max, mu_T * rho);
    
    if stopC_Q < tol_Q && stopC_P < tol_P
        break;
    end

end

%% Reshape X back to tensor form
X_tensor = reshape(X, [m, n, B]);

end


function [D_w, D_w_adj] = defDDt(w)
% Define weighted forward difference operator D_w and its adjoint D_w^*
    D_w     = @(U) ForwardD(U, w);
    D_w_adj = @(X, Y, Z) Dive(X, Y, Z, w);
end

function [D_h_U, D_v_U, D_t_U] = ForwardD(U, w)
% Forward weighted finite differences along horizontal, vertical, spectral modes
    B = size(U, 3);

    D_h_U = w(1) * [diff(U, 1, 2), U(:,1,:) - U(:,end,:)];
    D_v_U = w(2) * [diff(U, 1, 1); U(1,:,:) - U(end,:,:)];
    
    D_t_U = zeros(size(U));
    D_t_U(:,:,1:B-1) = w(3) * diff(U, 1, 3);
    D_t_U(:,:,B)     = w(3) * (U(:,:,1) - U(:,:,end));
end

function D_adj_U = Dive(X, Y, Z, w)
% Adjoint operator of weighted finite differences
    B = size(X, 3);

    D_adj_U = [X(:,end,:) - X(:,1,:), -diff(X,1,2)];
    D_adj_U = w(1) * D_adj_U + ...
              w(2) * [Y(end,:,:) - Y(1,:,:); -diff(Y,1,1)];

    Tmp = zeros(size(Z));
    Tmp(:,:,1)   = Z(:,:,end) - Z(:,:,1);
    Tmp(:,:,2:B) = -diff(Z,1,3);

    D_adj_U = D_adj_U + w(3) * Tmp;
end

function G = prox_L23(sigma, lambda)
% Proximal operator for ||G||_{2/3}^{2/3}
    x = 27 .* sigma .* sigma ./ (16 * lambda^(3/2));
    fai = acosh(x);
    Fai = (2 / sqrt(3)) * lambda^(1/4) .* (cosh(fai ./ 3)).^(1/2);
    h = abs(Fai) + sqrt(2 .* abs(sigma) ./ abs(Fai) - abs(Fai).^2);
    ht = (h.^3) ./ 8;
    G = (abs(sigma) > 48^(1/4) / 3 * lambda^(3/4)) .* sign(sigma) .* ht;
end

function [X, sv] = prox_nuclear(L, mu_Q, mu_P, B, sv, r)
% Proximal operator for nuclear norm with rank constraint
    if choosvd(B, sv) == 1
        [U, Sigma, V] = lansvd(L, sv, 'L');
    else
        [U, Sigma, V] = svd(L, 'econ');
    end

    Sigma = diag(Sigma);
    svp = min(length(find(Sigma > 1 / (mu_Q + mu_P))), r);

    if svp < sv
        sv = min(svp + 1, B);
    else
        sv = min(svp + round(0.05 * B), B);
    end

    X = U(:, 1:svp) * diag(Sigma(1:svp) - 1 / (mu_Q + mu_P)) * V(:, 1:svp)';
end